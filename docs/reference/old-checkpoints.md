# Old checkpoints

This page lists how a checkpoint saved before a feature existed loads and
behaves. {py:meth}`~dorothy.config.schema.ExperimentConfig.load_checkpoint_config`
fills in each missing setting, and
{py:func}`~dorothy.inference.checkpoint.load_checkpoint` rebuilds the model.

## Missing settings

A `config.yaml` without one of these settings loads with the value in the
second column. The model then predicts and evaluates as it did when it was
saved.

| Setting | Loads as | Training |
|---------|----------|----------|
| `data.normalize_spectra` | `false`, with a warning; see [below](#spectrum-normalization) | Allowed |
| `data.precision_encoding` | `false` | Rejected |
| `model.label_aware_uncertainty`, `multi_head_model.label_aware_uncertainty` | `false` | Rejected |
| `multi_head_model.residual_stream` | `false` | Allowed |

If one of these values is wrong for a checkpoint, add the setting to its
`config.yaml`.

## Spectrum normalization

A configuration without `data.normalize_spectra` loads as `true` when it
comes from the `dorothy_update` branch and does not set
`data.use_dense_loading: true`. A configuration comes from that branch when
it has a `multi_head_model.head_dropout` entry or 20 predicted parameters.
Every other configuration without the setting loads as `false`. Both cases
print a warning with the value used.

## Precision encoding

A checkpoint saved before the
[precision encoding](../methods/preprocessing.md) reads $\sigma$ in the error
channel of its inputs, not $\rho = \ln(1 + 1/\sigma)$.

- Prediction and evaluation give the spectra $\sigma$, as in its training.
- Saliency derivatives with respect to the error channel are derivatives
  with respect to $\sigma$.
- `dorothy train --resume` rejects the checkpoint.

## Label-aware uncertainty

A checkpoint saved before `label_aware_uncertainty` has the additive model.

| | Additive model (old) | Label-aware model |
|---|---|---|
| Output layer | One linear layer predicts $\mu$ and $\ln s$ | A label decoder predicts $\mu$, an error decoder $\ln s$ |
| What $s$ measures | The scatter beyond the reported label error $\sigma_y$ | The scatter around the label source's measurement, label noise included |
| Loss variance | $\sigma_y^2 + s^2$ | $s^2$ |
| z-score denominator in evaluation | $\sqrt{\sigma_y^2 + s^2}$; `HeadEvaluation.z_label_errors` holds $\sigma_y$ | $s$ |
| `dorothy info` | `additive` | `label-aware` |

- Prediction reports $s$, which excludes the label noise.
- The default embedding of an old MLP (`MLP.get_embeddings`,
  `AnomalyDetector`) is its last hidden layer.
- The checkpoint cannot be trained further: training rejects
  `label_aware_uncertainty: false`, and `--resume` needs the same layers.

## Residual stream

A multi-head checkpoint saved before `residual_stream` has plain blocks. Its
trunk has a 128-number output. The error decoder of each head concatenates
the log label errors to its input. `dorothy train --resume` needs the same
`residual_stream` value in the new configuration.

## Calibration factors

A checkpoint without `calibration.json` reports the raw predicted scatter $s$.
`dorothy info` prints `Calibration: none`.

## Head names

A multi-head model with one label source used to name its head `default`.
Its weights load into a head named after the first label source. Select the
head by that name: `--label-source apogee`, not `--label-source default`.

## Several label sources

| Checkpoint | Loads as | Evaluation |
|------------|----------|------------|
| `model` with several label sources | One head, the first label source, the one it trained on | Works |
| `multi_head_model`, one survey and several label sources | One head, the first label source, as saved | `DataSplit.load` raises `ValueError`; retrain the model |

The second kind trained only its first label source, and its split indexes
a data layout that this version no longer builds.

## Data split

| File | Behaviour |
|------|-----------|
| `data_split.pkl` of a dense layout without `index_space: rows` | Its indices point into the filtered arrays of the old loaders. `DataSplit.load` converts them to rows of the loaded arrays with `TrainingData.legacy_filtered_rows` |
| `data_split.pkl` of sparse storage without `index_space: rows` | Its indices are already catalogue rows and load unchanged |
| No `data_split.pkl` | `dorothy evaluate` recreates the split from the configured `seed` |

## Other settings

| Setting | Behaviour |
|---------|-----------|
| `training.learning_rate` | Becomes `training.scheduler.max_lr` for the `none`, `reduce_on_plateau` and `cosine_annealing` schedules; ignored with a warning for the others |
| A setting that this version does not have | Ignored with a warning |
| `multi_head_model.head_dropout` for a head the model does not have | Ignored with a warning |

## Weight files

| File | Behaviour |
|------|-----------|
| `final_model.pth` saved before it held the last epoch | Holds the best weights, like `best_model.pth` |
| A directory without `config.yaml`, such as an old epoch checkpoint | See [Weights without a configuration](#weights-without-a-configuration) |

## Weights without a configuration

`load_checkpoint` rebuilds a directory without `config.yaml` from the weights
alone.

- The model is an MLP with the additive output layer. Its layer sizes and
  normalization come from the weights, and its activation is the default,
  GELU.
- Label-aware weights raise `ValueError`: they need the configuration.
- The parameter names come from `normalizer.pkl`, or else are the first
  names of `STELLAR_PARAMETERS`.
- The scatter floor is 0.01, and there are no calibration factors.
- `dorothy predict` normalizes the spectra and does not apply the precision
  encoding.
