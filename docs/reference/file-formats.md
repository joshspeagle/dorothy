# File formats

This page lists the files that DOROTHY reads and writes: the super-catalogue,
the checkpoint directory, the predictions CSV and the FITS input of
`dorothy predict`.

## Super-catalogue

One HDF5 file, one row per star. [What is in the
super-catalogue](../catalogue/contents.md) explains the layout; this table
lists it. N is the number of stars and P the number of parameters.

| Path | Shape and type | Content |
|------|----------------|---------|
| `metadata/gaia_id` | (N,) int64 | Gaia DR3 source ID of each row, sorted |
| `metadata/ra`, `metadata/dec` | (N,) float64 | Degrees; 0 where unknown |
| `surveys/<survey>/flux` | (N, n_wavelengths) float32 | 0 where the survey has no spectrum |
| `surveys/<survey>/ivar` | (N, n_wavelengths) float32 | Inverse variance; 0 at masked or missing pixels |
| `surveys/<survey>/wavelength` | (n_wavelengths,) float32 | Å |
| `surveys/<survey>/snr` | (N,) float32 | Median S/N of the spectrum; 0 where missing |
| `surveys/lamost_mrs/spectra` | (N, 4, max(n_b, n_r)) float32 | flux_b, ivar_b, flux_r, ivar_r, each arm padded to the longer one |
| `surveys/lamost_mrs/wavelength_b`, `wavelength_r` | (n_b,), (n_r,) float32 | Å |
| `labels/<source>/values` | (N, P) float32 | 0 where missing |
| `labels/<source>/errors` | (N, P) float32 | 0 where missing |
| `labels/<source>/flags` | (N, P) uint8 | 0 = no warning |
| `labels/<source>/gaia_id` | (N,) int64 | Equal to `metadata/gaia_id` |

The file attributes are `n_stars`, `parameter_names`, `abundance_scale`,
`survey_names`, `label_sources`, `creation_date` and `version` (`2.0.0`).

## Checkpoint directory

`dorothy train` writes `<output_dir>/<name>/<name>_final/`.
{py:meth}`Trainer.save_checkpoint <dorothy.training.trainer.Trainer.save_checkpoint>`
writes the files without a note in the last column.

| File | Content | Note |
|------|---------|------|
| `best_model.pth` | PyTorch state dict of the epoch with the lowest validation loss | |
| `final_model.pth` | State dict of the last epoch | |
| `calibration.json` | Calibration factors of `best_model.pth`: label source → parameter → factor | Absent with `training.calibrate_uncertainties: false` or `loss: mse` |
| `config.yaml` | The completed configuration | |
| `normalizer.pkl` | Label normalization: parameter → `median`, `IQR`, and `log_median`, `log_IQR` for Teff | |
| `history_train_val.pkl` | Training history: a dict with `history_train`, `history_val`, `best_epoch` and the per-survey and per-label-source losses and metrics | |
| `learning_rates.pkl` | The learning rate of every optimizer step, as a list | |
| `data_split.pkl` | Training, validation and test catalogue rows (`train_idx`, `val_idx`, `test_idx`) | Written by `dorothy train` |
| `plots/` | Training plots | Written by `dorothy train` |
| `plots/evaluation/` | Evaluation plots | Written by `dorothy evaluate` |

With `training.ema_decay`, both weight files hold the averaged weights. The
`.pkl` files are pickles: load only files that you trust.

With `training.save_every: N`, training also writes
`<output_dir>/<name>/epoch_<N>/` every N epochs. Each holds
`model_epoch_<N>.pth` (the weights at that epoch),
`best_model_epoch_<N>.pth` (the best weights so far), `config.yaml`,
`normalizer.pkl`, `history_train_val.pkl` and `learning_rates.pkl`, but no
calibration factors. Pass the weights file to
{py:func}`~dorothy.inference.checkpoint.load_checkpoint` with `model_file`.

[Old checkpoints](old-checkpoints.md) lists how directories from older
versions load.

## Predictions CSV

`dorothy predict` writes one row per spectrum, in the order of the input
file.

| Column | Content |
|--------|---------|
| `star_id` | The ID of the spectrum (see [FITS input](#fits-input)) |
| `<parameter>` | The prediction, one column per parameter in the catalogue's order |
| `<parameter>_err` | Its uncertainty: the calibration factor times the predicted scatter, or the predicted scatter alone |
| `calibrated` | `True` when the uncertainties include calibration factors |

The values are in physical units, the units of the catalogue's labels, with
Teff in K. With `--normalized` they are in normalized units.

## FITS input

`dorothy predict` reads binary-table extensions by name, with one row per
spectrum.

| Extension | Column | Required | Content |
|-----------|--------|----------|---------|
| `SPECTRA` | `FLUX` | yes | Flux, one array of n_wavelengths per row |
| `SPEC_IVAR` | `IVAR` | yes | Inverse variance, same shape |
| `APOGEE` | `APOGEE_ID` or `TARGET_ID` | no | The star IDs |
| `GAIA_ID` | the first column | no | The star IDs, read when there is no `APOGEE` extension |
| `WAVELENGTH` | `WAVELENGTH` | no | The wavelength grid; not used for prediction |

- The spectra must be on the wavelength grid of the model's survey in the
  catalogue. The command checks only the number of bins.
- A pixel is valid when its flux is finite and its ivar is finite and
  positive; other pixels are masked.
- Without an ID column, `star_id` is the row number, from 0.
- The reader takes one `SPECTRA` extension, so it does not read LAMOST MRS
  files with one extension per arm.
