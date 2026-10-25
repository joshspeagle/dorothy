# Training

`dorothy train config.yaml` turns a configuration and the catalogue into a
{term}`checkpoint`. This page explains how training selects and splits the
stars, combines several label sources, warms up the loss, optimizes, averages
the weights and masks the data.

## Data selection and split

A star is used when it has a spectrum from at least one configured survey and
at least one valid label from a configured label source.
{py:func}`~dorothy.data.training_data.load_training_data` selects these stars
for training, for `dorothy evaluate` and for the Python API alike.

The split permutes the selected stars with
`numpy.random.RandomState(seed)`. Of n stars, the first
int(n · `train_ratio`) train, the next int(n · `val_ratio`) validate, and
the rest form the test set (70%, 20% and 10% by default). The trainer fits the
[label normalization](preprocessing.md#label-normalization) on the training
stars, chooses the best epoch and fits the
[calibration factors](uncertainties.md#calibration-factors) on the validation
stars, and leaves the test stars to `dorothy evaluate`.

## Several label sources

Each label source has its own {term}`head` and its own loss. With $S_k$ the
summed loss of the $n_k$ valid labels of source k in a batch and $w_k$ its
weight (`training.label_source_weights`, 1 by default),
`training.label_source_weighting` combines them in one of two ways:

$$
\mathcal{L}_\text{per\_label} = \frac{\sum_k w_k S_k}{\sum_k w_k n_k},
$$

$$
\mathcal{L}_\text{per\_source} = \frac{\sum_k w_k S_k / n_k}{\sum_k w_k} .
$$

`per_label`, the default, counts every valid label equally, so a source with
more labels in the batch weighs more. `per_source` counts each source's mean
loss equally, whatever its number of labels; it sums over the sources with at
least one valid label in the batch. The validation loss combines the heads in
the same way, and the training history also records each head's own best
epoch.

## Warm-up

The gradient $\partial\mathcal{L}/\partial\mu = 2(\mu - y)/s^2$ is small for
a label with a large s. Its residual then shrinks slowly, and s stays large
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

β falls linearly from 1 to 0 over the first W optimizer steps and then stays
0:

$$
\beta_t = \max\left(0,\ 1 - \frac{t}{W}\right),
$$

$$
W = \operatorname{round}(f\, n_\text{epochs}\, n_\text{steps}),
$$

where t counts the optimizer steps of the run from 0, $n_\text{steps}$ is the
number of steps per epoch and f is `training.nll_warmup_fraction`. f = 0
turns the warm-up off, and `loss: mse` has none.

::::{container} d-figure
```{raw} html
:file: ../figures/warmup.svg
:class: wide-only
```

```{raw} html
:file: ../figures/warmup-narrow.svg
:class: narrow-only
```

:::{container} caption
The warm-up: β falls from 1 to 0 over the first W optimizer steps, and the
plain loss trains the model after it.
:::
::::

The validation loss is always $\mathcal{L}$, so the best epoch, early stopping
and the `reduce_on_plateau` schedule compare all epochs on one objective.
Early stopping and `reduce_on_plateau` count only the epochs that end after
the warm-up, because a plateau of $\mathcal{L}$ during the warm-up does not
show convergence. The training losses in the history are those of
$\mathcal{L}_\beta$, so during the warm-up they do not compare with the
validation losses. The loss plot shades the warm-up; the epoch in which it
ends mixes both losses.

## Optimization

The default optimizer is AdamW {cite:p}`loshchilov2019adamw`, with weight
decay on the weight matrices only: biases and normalization weights are not
decayed. The default learning-rate schedule is one cycle
{cite:p}`smith2019superconvergence`, which also cycles the momentum (β₁ of
Adam and AdamW, or the momentum of SGD) between 0.85 and 0.95, opposite to
the learning rate. Gradients are clipped to a norm of `training.gradient_clip`.
The batches are shuffled by a generator seeded by `seed`.

With mixed precision (`training.amp`, CUDA only), the training forward pass
runs under autocast in `training.amp_dtype`. The loss and validation run in
float32, so validation gives the metrics that `dorothy evaluate` reproduces.
bfloat16 has the exponent range of float32 and needs no gradient scaler.
float16 has a smaller range, so a gradient scaler keeps small gradients from
underflowing to zero.

After each epoch the model is evaluated on the validation stars. The epoch
with the lowest validation loss is the best epoch. With
`training.early_stopping_patience`, training stops after that many epochs
without a lower validation loss after the warm-up. If no epoch has a finite
validation loss, there is no best epoch, and training raises `RuntimeError`.
Batch normalization needs at least two stars per batch in training.

## Weight averaging

`training.ema_decay: d` keeps an exponential moving average (EMA) of the
weights during training. After each optimizer step, the trained weights $w$
update the average $\bar{w}$:

$$
\bar{w} \leftarrow d\,\bar{w} + (1 - d)\,w .
$$

The average starts from the weights before the first step. It includes the
buffers, such as the running statistics of batch normalization; integer
buffers are copied, not averaged. Validation uses the averaged weights, and so
do the choice of the best epoch, early stopping, the `reduce_on_plateau`
schedule and every saved weight file. After training the model holds the
averaged weights of the best epoch. The training steps, and the weight norms
and updates in the history, use the trained weights.

The average lags the trained weights: the weights from n steps before have the
share $(1 - d)\,d^n$, a mean age of $d/(1 - d)$ steps (999 for d = 0.999).
Averaging the weights along the training trajectory can improve
generalization {cite:p}`izmailov2018swa`, and it makes the result depend less
on the exact stopping epoch. It is off by default, because it also lags a
head that starts to overfit.

The average is one more copy of the weights on the training device, 4 bytes
per parameter. Training with AdamW already holds five such copies: the
weights, their gradients, the two moment estimates and the weights of the
best epoch. If memory on the training device is short,
`training.ema_cpu_every: k` keeps the average in CPU memory. It then updates
after the first optimizer step and every k steps after that, with the decay
$d^k$. Over k steps, an average updated at every step also keeps the fraction
$d^k$ of its old value, so both forget at the same rate.

## Augmentation

Two optional augmentations hide parts of each batch. They teach the model to
predict from incomplete inputs, as it must for stars that only some surveys
observed.

::::{container} d-figure
```{raw} html
:file: ../figures/masking.svg
:class: wide-only
```

```{raw} html
:file: ../figures/masking-narrow.svg
:class: narrow-only
```

:::{container} caption
Input masking hides surveys and blocks of pixels, and label masking hides
label sources and labels; the batch is illustrative.
:::
::::

`input_masking`
: Keeps each survey of a star with a probability drawn from
  [`p_survey_min`, `p_survey_max`]. A hidden survey is 0 in all three channels
  and counts as missing. Within each kept survey it hides blocks of pixels.
  The block size is a fraction f of the N pixels of the spectrum, with ln f
  drawn uniformly between ln(1/N), one pixel, and ln `f_max`. A random offset
  moves the
  block boundaries, so no pixel always sits at an edge. Each block is kept
  with a probability drawn from [`p_block_min`, `p_block_max`]. With one
  survey, only blocks are hidden.

`label_masking`
: Keeps each label source of a star with a probability drawn from
  [`p_labelset_min`, `p_labelset_max`], and each label of a kept source with a
  probability drawn from [`p_label_min`, `p_label_max`].

The probabilities, the block size and the offset are drawn again for every
batch, and the block size and offset for every survey. Each star keeps at
least one survey, one block, one label source and one label. The masks come
from generators seeded by `seed`, so runs with the same seed on the same
device mask the same entries.

## Checkpoints

A run writes the checkpoint directory `<output_dir>/<name>/<name>_final/`:

| File | Contents |
|------|----------|
| `best_model.pth` | Weights from the best epoch |
| `final_model.pth` | Weights after the last epoch |
| `calibration.json` | One factor per head and parameter that scales the predicted scatter of `best_model.pth` |
| `config.yaml` | The completed configuration |
| `normalizer.pkl` | The label normalization |
| `data_split.pkl` | Rows of the training, validation and test sets |
| `history_train_val.pkl` | Losses and metrics per epoch |
| `learning_rates.pkl` | The learning rate after each optimizer step |
| `plots/` | Training plots; the bar charts of validation metrics show the best epoch |

`data_split.pkl` stores catalogue rows for the sparse layout and rows of the
loaded arrays for the dense layouts. `training.save_every: N` also saves the
weights, the configuration and the normalization every N epochs, in
`epoch_N/`, without calibration factors. The weights there are
`model_epoch_N.pth` and `best_model_epoch_N.pth`; `dorothy evaluate --model`
loads them, and `dorothy predict` cannot.
`dorothy train config.yaml --resume <checkpoint>` starts from the weights of
an earlier run, with a new optimizer, schedule and warm-up.
