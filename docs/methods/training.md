# Training

`dorothy train config.yaml` runs these steps:

1. Complete the configuration from the catalogue: the survey wavelength
   counts, the output heads and the parameter names
   ({py:func}`~dorothy.data.training_data.complete_config_from_catalogue`).
2. Load the spectra and labels
   ({py:func}`~dorothy.data.training_data.load_training_data`).
3. Split the stars into training, validation and test sets.
4. Fit the label normalization on the training stars, train, and fit the
   [calibration factors](evaluation.md#calibration-factors) of the predicted
   scatter ({py:class}`~dorothy.training.trainer.Trainer`).
5. Save the {term}`checkpoint`, the split and the training plots.

`dorothy evaluate` and the Python API load and split data through the same
functions, so evaluation sees exactly the stars and inputs that training saw.

## Data selection and split

A star is used when it has a spectrum from at least one configured survey and
at least one valid label from a configured label source. The split is a
permutation of these stars drawn with `numpy.random.RandomState(seed)`: the
first `data.train_ratio` of them train, the next `data.val_ratio` validate,
and the rest form the test set (70%, 20% and 10% by default). The checkpoint
stores the catalogue rows of each set in `data_split.pkl`.

## The loss

For one valid label with value y, the heteroscedastic loss is the negative
log-likelihood of y under a normal distribution centred on the prediction μ
with standard deviation s, up to a constant and a factor of 2:

$$
\mathcal{L} = \frac{(\mu - y)^2}{s^2} + \ln s^2,
\qquad s^2 = e^{2 \ln s} + s_0^2 .
$$

The first term rewards accurate predictions, weighted by the predicted
variance; the second keeps the model from inflating its scatter s. The scatter
floor s₀ (`training.scatter_floor`, 0.01 in normalized units by default) stops
s from collapsing to zero. The model output ln s is clamped to [−10, 5] before
use. Masked labels contribute nothing.
{py:func}`~dorothy.losses.heteroscedastic.predicted_scatter` is the one
definition of s, shared by the loss, validation, evaluation and prediction.

### Scatter and label errors

Every head is a {py:class}`~dorothy.models.output.LabelAwareOutput` with two
parallel decoders that read the same head input x. The label decoder D_μ
predicts μ from x alone. The error decoder D_s predicts ln s from x and, for
each parameter p, the label source's reported error σ_y,p in normalized units:

$$
\mu = D_\mu(x), \qquad
\ln s = D_s(x, u), \qquad
u_p = \begin{cases}
\ln \sigma_{y,p} & \text{if } \sigma_{y,p} > 0, \\
m_p & \text{otherwise,}
\end{cases}
$$

where ln σ_y,p is clamped to [−10, 5] like ln s, and m is the learned
{term}`missing vector`. m starts at 0, an error of one normalized unit, far
above real label errors. No layer that feeds μ reads u: the label errors and
the missing labels correlate with the label values, and prediction has no
label errors.

```{raw} html
:file: ../figures/uncertainty.svg
```

Each decoder is a stack of hidden layers (linear, normalization, activation
and optional dropout) followed by a linear layer with one output per
parameter. Both decoders have the same hidden layer sizes, and u enters the
first layer of the error decoder:

Multi-head model
: x is the trunk output, and the hidden layers are the `output_hidden`
  layers.

MLP
: The decoders take over the last hidden layer: x is the input of that
  layer, and each decoder has one hidden layer of its size. μ therefore
  passes through exactly the layers of `model.hidden_layers`.

The error decoder always has a hidden layer, so that ln s can be a nonlinear
function of u. If the true label noise is $\sqrt{a}\,\sigma_y$ and adds in
quadrature to the model error $\varepsilon$, for example,

$$
s^2 = a \sigma_y^2 + \varepsilon^2, \qquad
\ln s = \tfrac{1}{2} \ln\left(a\, e^{2u} + \varepsilon^2\right),
$$

which tends to ln ε for small label errors and to u + ½ ln a for large ones.

In training, masked labels always get m. A fraction
`training.label_error_dropout` of the stars (0.5 by default), drawn afresh for
every batch and head, also get m for all their parameters, so the model learns
to predict without the errors. Validation, evaluation and prediction always
use m, so the best epoch is the best one for the reported uncertainties.

The predicted scatter s is therefore the expected scatter between the
prediction and the label source's measurement of the star, the source's label
noise included. The variance s² has no term fixed by the reported errors, so s
follows the scatter that the labels show: for a label source that
overestimates its errors, s falls below them.

Checkpoints saved before `label_aware_uncertainty` existed have the additive
model: one linear layer predicts μ and ln s, the loss variance is σ_y² + s²,
and s is the scatter beyond the reported label error. Their configuration has
no `label_aware_uncertainty`, so it loads as false: they predict and evaluate
exactly as before. They cannot be trained further, because training rejects
`label_aware_uncertainty: false` and `--resume` needs the same layers.

`loss: mse` trains μ alone with a masked mean squared error and leaves ln s
untrained.

### Several label sources

With several label sources, each head has its own loss, and
`training.label_source_weighting` sets how they combine:

`per_label` (default)
: Every valid label in the batch counts equally. A label source with more
  labels in the batch weighs more.

`per_source`
: Each head's mean loss over its valid labels counts equally, whatever the
  number of labels.

`training.label_source_weights` multiplies each source's contribution, for
example `{apogee: 1.0, galah: 0.5}`.

### Warm-up

The gradient $\partial\mathcal{L}/\partial\mu = 2(\mu - y)/s^2$ is small for
a label with a large s, so its residual shrinks slowly and s stays large
{cite:p}`seitzer2022pitfalls`. Training therefore starts with the β-NLL loss
of that paper, which multiplies the loss of each label by $s^{2\beta}$:

$$
\mathcal{L}_\beta = \operatorname{sg}\left(s^{2\beta}\right)
\left[\frac{(\mu - y)^2}{s^2} + \ln s^2\right],
$$

where sg (stop-gradient) passes the value of its argument but no gradient.
For β = 1, $\partial\mathcal{L}_\beta/\partial\mu = 2(\mu - y)$, the gradient
of the squared error, while s still learns from the residuals. For β = 0,
$\mathcal{L}_\beta = \mathcal{L}$.

β falls linearly from 1 to 0 over the first W optimizer steps of the run and
then stays 0:

$$
\beta_t = \max\left(0,\ 1 - \frac{t}{W}\right), \qquad
W = \operatorname{round}(f\, n_\text{epochs}\, n_\text{steps}),
$$

where t counts the optimizer steps from 0, $n_\text{steps}$ is the number of
steps per epoch and f is `training.nll_warmup_fraction` (0.1 by default; 0
turns the warm-up off). `loss: mse` has no warm-up.

The validation loss is always $\mathcal{L}$, so the best epoch, early stopping
and the `reduce_on_plateau` schedule compare all epochs on one objective.
Early stopping and `reduce_on_plateau` count only the epochs that end after the
warm-up. During the warm-up the model trains on $\mathcal{L}_\beta$, so a
plateau of $\mathcal{L}$ does not show convergence. The training losses in the
history are those of $\mathcal{L}_\beta$, so during the warm-up they are not
comparable with the validation losses. The loss plot shades the warm-up.

## Optimization

The default optimizer is AdamW {cite:p}`loshchilov2019adamw` (`adam` and
`sgd` are also available) with a one-cycle learning-rate schedule
{cite:p}`smith2019superconvergence` (also `cyclic`, `reduce_on_plateau`,
`cosine_annealing` or `none`). The one-cycle schedule also cycles the
momentum (the β₁ of Adam and AdamW, or SGD's momentum) between 0.85 and 0.95,
opposite to the learning rate. Gradients are clipped to a norm of
`training.gradient_clip` (10 by default).

With mixed precision (`training.amp`, CUDA only), the training forward pass
runs under autocast in `training.amp_dtype`. The loss and validation run in
float32, so validation gives the metrics that `dorothy evaluate` reproduces.
bfloat16 has the exponent range of float32 and needs no gradient scaler.
float16 has a smaller range, so a gradient scaler keeps small gradients from
underflowing to zero.

After each epoch the model is evaluated on the validation stars. The epoch
with the lowest validation loss is the best epoch; with several label sources
the validation loss combines the heads as training does, and the history
also records each head's own best epoch. Training stops early after
`training.early_stopping_patience` epochs without improvement after the
[warm-up](#warm-up), if set. If no epoch has a finite validation loss, there
is no best epoch, and training raises `RuntimeError`.

## Weight averaging

`training.ema_decay: d` keeps an exponential moving average (EMA) of the
weights during training. After each optimizer step, the trained weights $w$
update the average $\bar{w}$:

$$
\bar{w} \leftarrow d\,\bar{w} + (1 - d)\,w .
$$

The average starts from the weights before the first step. It includes the
buffers, such as the running statistics of batch normalization; integer
buffers are copied, not averaged. Validation, and so the choice of the best
epoch, early stopping and the `reduce_on_plateau` schedule, use the averaged
weights, as do `best_model.pth`, `final_model.pth` and the weights saved
every `training.save_every` epochs. After training the model holds the
averaged weights of the best epoch. The training steps, and the weight norms
and updates in the history, use the trained weights.

The average is one more copy of the weights on the training device, 4 bytes
per parameter. Training with AdamW already holds five such copies: the
weights, their gradients, the two moment estimates and the weights of the
best epoch. If memory on the training device is short,
`training.ema_cpu_every: k` keeps the average in CPU memory. Each update then
copies the weights to CPU memory, so it runs after the first optimizer step
and every k steps after that, with the decay $d^k$: over k steps, an average
updated at every step also keeps the fraction $d^k$ of its old value, so both
forget at the same rate.

Averaging the weights along the training trajectory can improve
generalization {cite:p}`izmailov2018swa`, and it makes the result depend less
on the exact stopping epoch. The average lags the trained weights: the weights
from n steps before have the share $(1 - d)\,d^n$, a mean age of $d/(1 - d)$
steps (999 for d = 0.999). It is off by default (`ema_decay: null`) because
it also lags a head that starts to overfit.

## Augmentation

Two optional augmentations hide parts of each batch, with probabilities drawn
afresh for every batch from the configured ranges.

```{raw} html
:file: ../figures/masking.svg
```

`input_masking`
: Hides whole surveys of a star, each kept with a probability drawn from
  [`p_survey_min`, `p_survey_max`]; a hidden survey counts as missing. Within
  each kept survey, it hides blocks of pixels: block sizes are drawn
  log-uniformly up to a fraction `f_max` of the spectrum, and each block is
  kept with a probability drawn from [`p_block_min`, `p_block_max`]. The block
  boundaries move with a random offset, so no pixel always sits at an edge.

`label_masking`
: Hides label sources and individual labels in the same way
  (`p_labelset_*`, `p_label_*`).

Each star always keeps at least one survey, one block, one label source and
one label. Masking teaches the model to predict from incomplete inputs, as it
must for stars observed by only some surveys.

## Checkpoints

A run writes `<output_dir>/<name>/<name>_final/`:

| File | Contents |
|------|----------|
| `best_model.pth` | Weights from the best epoch |
| `final_model.pth` | Weights after the last epoch |
| `calibration.json` | One factor per head and parameter that scales the predicted scatter of `best_model.pth` |
| `config.yaml` | The completed configuration |
| `normalizer.pkl` | The label normalization |
| `data_split.pkl` | Catalogue rows of the training, validation and test sets |
| `history_train_val.pkl` | Losses and metrics per epoch |
| `learning_rates.pkl` | The learning rate after each optimizer step |
| `plots/` | Training plots; the bar charts of validation metrics show the best epoch |

`dorothy train config.yaml --resume <checkpoint>` starts from the weights of
an earlier run, with a new optimizer, schedule and warm-up.
`training.save_every: N` also saves the weights, the configuration and the
normalization every N epochs, in `epoch_N/`, without calibration factors.
BatchNorm needs at least two stars per batch in training.
