# Training

`dorothy train config.yaml` runs these steps:

1. Complete the configuration from the catalogue: the survey wavelength
   counts, the output heads and the parameter names
   ({py:func}`~dorothy.data.training_data.complete_config_from_catalogue`).
2. Load the spectra and labels
   ({py:func}`~dorothy.data.training_data.load_training_data`).
3. Split the stars into training, validation and test sets.
4. Fit the label normalization on the training stars, train, and fit the
   [calibration factors](uncertainties.md#calibration-factors) of the predicted
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
stores the rows of each set in `data_split.pkl`: catalogue rows for sparse
storage, rows of the loaded arrays for the dense layouts.

## Optimization

The default optimizer is AdamW {cite:p}`loshchilov2019adamw` (`adam` and
`sgd` are also available) with a one-cycle learning-rate schedule
{cite:p}`smith2019superconvergence` (also `cyclic`, `reduce_on_plateau`,
`cosine_annealing` or `none`); `training.scheduler.max_lr` sets the learning
rate of each. The one-cycle schedule also cycles the momentum (the β₁
of Adam and AdamW, or SGD's momentum) between 0.85 and 0.95, opposite to the
learning rate. Gradients are clipped to a norm of `training.gradient_clip` (10
by default).

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
[warm-up](uncertainties.md#warm-up), if set. If no epoch has a finite validation loss, there
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
The masks come from generators seeded by `seed`, so runs with the same seed on
the same device mask the same entries.

## Checkpoints

A run writes `<output_dir>/<name>/<name>_final/`:

| File | Contents |
|------|----------|
| `best_model.pth` | Weights from the best epoch |
| `final_model.pth` | Weights after the last epoch |
| `calibration.json` | One factor per head and parameter that scales the predicted scatter of `best_model.pth` |
| `config.yaml` | The completed configuration |
| `normalizer.pkl` | The label normalization |
| `data_split.pkl` | Rows of the training, validation and test sets ([Data selection and split](#data-selection-and-split)) |
| `history_train_val.pkl` | Losses and metrics per epoch |
| `learning_rates.pkl` | The learning rate after each optimizer step |
| `plots/` | Training plots; the bar charts of validation metrics show the best epoch |

`dorothy train config.yaml --resume <checkpoint>` starts from the weights of
an earlier run, with a new optimizer, schedule and warm-up.
`training.save_every: N` also saves the weights, the configuration and the
normalization every N epochs, in `epoch_N/`, without calibration factors.
The weights there are `model_epoch_N.pth` and `best_model_epoch_N.pth`.
`dorothy evaluate --model` loads them; `dorothy predict` cannot.
BatchNorm needs at least two stars per batch in training.
