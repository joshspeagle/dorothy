# Training

`dorothy train config.yaml` runs these steps:

1. Complete the configuration from the catalogue: the survey wavelength
   counts, the output heads and the parameter names
   ({py:func}`~dorothy.data.training_data.complete_config_from_catalogue`).
2. Load the spectra and labels
   ({py:func}`~dorothy.data.training_data.load_training_data`).
3. Split the stars into training, validation and test sets.
4. Fit the label normalization on the training stars and train
   ({py:class}`~dorothy.training.trainer.Trainer`).
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

For one valid label with value y and uncertainty σ_y, the heteroscedastic loss
is the negative log-likelihood of y under a normal distribution centred on the
prediction μ, up to a constant and a factor of 2:

$$
\mathcal{L} = \frac{(\mu - y)^2}{\sigma_y^2 + s^2} + \ln\left(\sigma_y^2 + s^2\right),
\qquad s^2 = e^{2 \ln s} + s_0^2 .
$$

```{raw} html
:file: ../figures/uncertainty.svg
```

The first term rewards accurate predictions, weighted by the total variance;
the second keeps the model from inflating its scatter s. The scatter floor s₀
(`training.scatter_floor`, 0.01 in normalized units by default) stops s from
collapsing to zero for labels with small reported uncertainties. The model
output ln s is clamped to [−10, 5] before use. Masked labels contribute
nothing. {py:func}`~dorothy.losses.heteroscedastic.predicted_scatter` is the
one definition of s, shared by the loss, validation, evaluation and
prediction.

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

## Optimization

The default optimizer is AdamW {cite:p}`loshchilov2019adamw` (`adam` and
`sgd` are also available) with a one-cycle learning-rate schedule
{cite:p}`smith2019superconvergence` (also `cyclic`, `reduce_on_plateau`,
`cosine_annealing` or `none`). Gradients are clipped to a norm of
`training.gradient_clip` (10 by default). `training.amp: true` trains with
mixed precision on CUDA; it is off by default because it produced NaN losses
in multi-survey training.

After each epoch the model is evaluated on the validation stars. The epoch
with the lowest validation loss is the best epoch; with several label sources
the validation loss combines the heads as training does, and the history
also records each head's own best epoch. Training stops early after
`training.early_stopping_patience` epochs without improvement, if set. If no
epoch has a finite validation loss, there is no best epoch, and training
raises `RuntimeError`.

## Weight averaging

`training.ema_decay: d` keeps an exponential moving average (EMA) of the
weights during training. After each optimizer step, the trained weights $w$
update the average $\bar{w}$:

$$
\bar{w} \leftarrow d\,\bar{w} + (1 - d)\,w .
$$

The average starts from the weights before the first step. It includes the
buffers, such as the running statistics of batch normalization; integer
buffers are copied, not averaged. Validation, the choice of the best epoch,
`best_model.pth`, `final_model.pth` and the weights saved every
`training.save_every` epochs use the averaged weights, and after training the
model holds the averaged weights of the best epoch. The training steps, and
the weight norms and updates in the history, use the trained weights.

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
| `config.yaml` | The completed configuration |
| `normalizer.pkl` | The label normalization |
| `data_split.pkl` | Catalogue rows of the training, validation and test sets |
| `history_train_val.pkl` | Losses and metrics per epoch |
| `learning_rates.pkl` | The learning rate after each optimizer step |
| `plots/` | Training plots; the bar charts of validation metrics show the best epoch |

`dorothy train config.yaml --resume <checkpoint>` starts from the weights of
an earlier run, with a new optimizer and schedule. `training.save_every: N`
also saves the weights every N epochs. BatchNorm needs at least two stars per
batch in training.
