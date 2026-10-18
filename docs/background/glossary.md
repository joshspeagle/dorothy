# Glossary

```{glossary}
:sorted:

checkpoint
  The directory a training run writes: the best and final weights, the
  calibration factors, the configuration, the label normalization and the
  data split. See [Training](../methods/training.md#checkpoints).

encoder
  The part of a multi-head model that maps one survey's spectrum to a vector
  of fixed length (`latent_dim`). Each survey has its own. See
  [Models](../methods/models.md).

flag
  A quality flag of one label, 0 when the label source raises no warning. A
  label with a nonzero flag is masked. `data.max_flag_bits` also masks every
  label of a star with too many flagged parameters.

has_data
  A boolean per star and survey: whether the star has a spectrum from that
  survey in the batch. Missing surveys are skipped by the model.

head
  The part of a multi-head model that predicts the labels of one label source
  from the trunk's output. See [Models](../methods/models.md).

label
  One stellar parameter or abundance of one star, as measured by a label
  source, with its uncertainty and flag. The parameters are set by the
  catalogue, for example `teff`, `logg`, `fe_h` and `mg_fe`.

label source
  A high-resolution survey whose labels are the training targets: APOGEE or
  GALAH.

mask
  A channel that marks valid entries with 1 and invalid ones with 0: valid
  pixels in a spectrum ({term}`ivar` > 0), valid labels in a label array.
  Masked entries can hold any value and are never used.

missing vector
  The learned input of a head's log-scatter that stands in for a missing
  label error. See [Training](../methods/training.md#scatter-and-label-errors).

ivar
  The inverse variance of a flux measurement, 1/σ². A pixel with ivar = 0
  carries no information.

normalized units
  The units the model predicts in: each parameter minus its training median,
  divided by its training interquartile range, with Teff in log10. See
  [Preprocessing](../methods/preprocessing.md#labels).

predicted scatter
  The model's estimate s of the spread of the measured labels around its
  prediction, label noise included: s = sqrt(exp(2 ln s) + s₀²), where ln s
  is the model output and s₀ the {term}`scatter floor`.

scatter floor
  The smallest predicted scatter, s₀ (`training.scatter_floor`, in normalized
  units).

split
  The division of the usable stars into training, validation and test sets,
  stored in the checkpoint as catalogue rows.

super-catalogue
  The HDF5 file with every star's spectra and labels, one row per star. See
  [The super-catalogue](../methods/catalogue.md).

survey
  A spectroscopic survey whose spectra are model inputs: DESI, BOSS, LAMOST
  LRS or LAMOST MRS.

trunk
  The part of a multi-head model that combines the encoders' vectors and
  passes its output to every head.

z-score
  The residual of a prediction divided by its predicted scatter:
  z = (μ − y) / s. See [Calibration](../methods/evaluation.md#calibration).
```
