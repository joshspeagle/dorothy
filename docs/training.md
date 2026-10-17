# Training

`dorothy train config.yaml` runs these steps, in this order:

1. Complete the configuration from the catalogue
   ([`complete_config_from_catalogue`][dorothy.data.training_data.complete_config_from_catalogue]).
2. Load the spectra and labels
   ([`load_training_data`][dorothy.data.training_data.load_training_data]).
3. Split the stars into training, validation and test sets.
4. Fit the label normalization on the training stars and train
   ([`Trainer`][dorothy.training.trainer.Trainer]).
5. Save the checkpoint, the split and the training plots.

`dorothy evaluate` and the Python API load data through the same functions,
so evaluation sees exactly the inputs that training saw.

## Data

### Spectra

A spectrum enters the model as an array of shape `(3, n_wavelengths)`:

| Channel | Content |
|---------|---------|
| 0 | flux |
| 1 | sigma = 1/sqrt(ivar), 0 where ivar <= 0 |
| 2 | mask: 1 where ivar > 0, else 0 |

With `data.normalize_spectra: true` (the default), each spectrum is first
scaled by its own median and interquartile range (IQR) over valid pixels:
flux becomes (flux - median) / IQR and ivar becomes IQR² · ivar. Pixels with
non-finite flux or ivar <= 0 are invalid and get flux = ivar = 0.

LAMOST MRS spectra have two arms. The model input concatenates the blue arm
and the red arm, each on its own wavelength grid.

### Labels

Labels have shape `(3, n_parameters)`: values, errors and a mask. A label is
valid when its flag is 0 and its error is positive. With
`data.max_flag_bits >= 0`, a star with more flagged parameters than that
loses all its labels. Masked entries can hold any value and are never used.

### Selection and split

A star is used when it has a spectrum from at least one configured survey and
at least one valid label. The split is a permutation of these stars with
`numpy.random.RandomState(seed)`: the first `train_ratio` of them train, the
next `val_ratio` validate, and the rest are the test set (70/20/10 by
default). `data_split.pkl` stores the catalogue rows of each set.

### Storage

With several surveys, the default sparse storage keeps each survey's spectra
only for the stars that have them
([`SparseMergedData`][dorothy.data.catalogue_loader.SparseMergedData]), and
each batch is assembled on the fly. `data.use_dense_loading: true` instead
stores a full array per survey, zeros included, which needs much more memory.

## Label normalization

[`LabelNormalizer`][dorothy.data.normalizer.LabelNormalizer] scales each
parameter by its median and IQR over the training stars; Teff is normalized
in log10. With several label sources, it is fit on all of them: where two
sources label the same star, the first source in `data.label_sources` is
used. The model predicts in normalized units;
[`inverse_transform`][dorothy.data.normalizer.LabelNormalizer.inverse_transform]
converts predictions and uncertainties back to physical units.

## Models

`model` configures an [`MLP`][dorothy.models.mlp.MLP]: the flattened
`(3, n_wavelengths)` input passes through the hidden layers to 2 ×
`n_parameters` outputs.

`multi_head_model` configures a
[`MultiHeadMLP`][dorothy.models.multi_head_mlp.MultiHeadMLP]:

```
BOSS spectrum ─> BOSS encoder ─┐                ┌─> APOGEE head
                               ├─> shared trunk ┤
DESI spectrum ─> DESI encoder ─┘                └─> GALAH head
```

- Each encoder maps its survey's spectrum to a vector of `latent_dim`.
- `combination_mode: concat` places the survey vectors side by side, with
  zeros for surveys a star lacks. `mean` averages the vectors of the surveys
  the star has.
- The trunk processes the combined vector; each head predicts the
  parameters of its label source from the trunk output.

Both models output an array of shape `(2, n_parameters)` per star: the
predicted value μ and the log-scatter ln s, in normalized units.

## Loss

The heteroscedastic loss of one valid label is

$$
\frac{(\mu - y)^2}{\sigma_y^2 + s^2} + \ln\left(\sigma_y^2 + s^2\right),
\qquad s^2 = e^{2 \ln s} + s_0^2,
$$

where y is the label, σ_y its error and s_0 the scatter floor
(`training.scatter_floor`). The predicted scatter s is the model's estimate
of the scatter in addition to the label error. ln s is clamped to [-10, 5]
before use. `loss: mse` trains μ alone with a masked mean squared error.

### Several label sources

`training.label_source_weighting` sets how the heads' losses combine:

- `per_label` (default): every valid label in the batch counts equally, so a
  label source with more labels has more weight.
- `per_source`: each head's mean loss counts equally.

`training.label_source_weights` multiplies each source's contribution. The
validation loss, which selects the best epoch, combines the heads the same
way. The training history also records each head's own best epoch.

## Augmentation

`input_masking` hides parts of the input during training, freshly for each
batch:

- whole surveys of a star, each kept with a probability drawn from
  [`p_survey_min`, `p_survey_max`]; a hidden survey counts as missing;
- blocks of pixels within each kept survey. Block sizes are drawn
  log-uniformly up to `f_max` of the spectrum, and each block is kept with a
  probability drawn from [`p_block_min`, `p_block_max`].

`label_masking` hides label sources and individual labels in the same way.
At least one survey, one block, one label source and one label per star are
always kept.

## Checkpoints

A run writes `<output_dir>/<name>/<name>_final/`:

| File | Contents |
|------|----------|
| `best_model.pth` | Weights from the epoch with the lowest validation loss |
| `final_model.pth` | Weights after the last epoch |
| `config.yaml` | The completed configuration |
| `normalizer.pkl` | The label normalization |
| `data_split.pkl` | Catalogue rows of the training, validation and test sets |
| `history_train_val.pkl` | Losses and metrics per epoch |
| `learning_rates.pkl` | Learning rate per optimizer step |
| `plots/` | Training plots |

`dorothy train config.yaml --resume <checkpoint_dir>` starts from the weights
of an earlier run, with a new optimizer and learning-rate schedule.
`training.save_every: N` also saves the weights every N epochs.

## Notes

- BatchNorm (`normalization: batchnorm`) needs at least two stars per batch
  in training.
- `training.amp: true` trains with mixed precision on CUDA. It is off by
  default because it produced NaN losses in multi-survey training.
