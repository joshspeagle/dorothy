# Evaluation and prediction

## Evaluate a checkpoint

```bash
dorothy evaluate <checkpoint_dir>
dorothy evaluate <checkpoint_dir> --format json --output results.json
dorothy evaluate <checkpoint_dir> --model final_model.pth --no-plots
```

`evaluate` reloads the catalogue with the checkpoint's configuration, takes
the test stars from `data_split.pkl` and scores every output head:

- **Normalized units:** the units the model predicts in. On the validation
  stars these metrics equal the training validation metrics exactly.
- **Physical units:** K for Teff, dex for log g and abundances.

It also writes four plots per label source to
`<checkpoint>/plots/evaluation/`: predictions against labels, residuals,
histograms of predictions and labels, and z-score histograms. The JSON
output holds the primary head's metrics at the top level and every head's
metrics under `label_sources`.

From Python, [`evaluate_model`][dorothy.inference.evaluation_utils.evaluate_model]
returns one [`HeadEvaluation`][dorothy.inference.evaluation_utils.HeadEvaluation]
per label source, with the metrics and the predictions in physical units.

## Metrics

For each parameter, over the stars with a valid label, with residual
r = prediction - label:

| Metric | Definition |
|--------|------------|
| `rmse` | sqrt(mean(r²)) |
| `bias` | mean(r) |
| `sd` | standard deviation of r |
| `mae` | mean(abs(r)) |
| `median_offset` | median(r) |
| `robust_scatter` | (P84 - P16) / 2 of r |
| `z_median` | median of z |
| `z_robust_scatter` | (P84 - P16) / 2 of z |
| `pred_unc_p16`, `pred_unc_p50`, `pred_unc_p84` | percentiles of the predicted scatter s |

The z-score z = r / sqrt(σ_y² + s²) combines the label error σ_y and the
predicted scatter s (see [Training](training.md#loss)). For calibrated
uncertainties, `z_robust_scatter` is close to 1; above 1, the model
underestimates its uncertainties. `Evaluator.evaluate_by_survey` gives the
same metrics for the stars of each survey.

## Predict for new spectra

```bash
dorothy predict --checkpoint <checkpoint_dir> --input spectra.fits \
    --output predictions.csv --survey boss --label-source galah
```

The input is a FITS file with `SPECTRA` (column `FLUX`) and `SPEC_IVAR`
(column `IVAR`) table extensions, one row per spectrum, on the wavelength
grid the model was trained on. An optional `WAVELENGTH` extension (column
`WAVELENGTH`) and `GAIA_ID` or `APOGEE` extension provide the grid and the
star IDs. LAMOST MRS files with two arms are not supported yet.

`predict` applies the training preprocessing (including the per-spectrum
normalization when the model used it) and writes one CSV row per spectrum,
with each parameter and its uncertainty (`teff`, `teff_err`, ...) in
physical units. The uncertainty is the predicted scatter s. Options:

- `--survey`: the survey of the spectra, required for multi-survey models.
- `--label-source`: the output head to use (default: the first label source).
- `--normalized`: write normalized units instead of physical units.

From Python, [`Predictor`][dorothy.inference.predictor.Predictor] predicts
for arrays of shape `(n, 3, n_wavelengths)`, or for a dict of such arrays per
survey together with `has_data`; `get_embeddings` returns the model's latent
vectors.

## Inspect a checkpoint

```bash
dorothy info <checkpoint_dir>
```

`info` prints the model architecture, the configuration, the parameter names
and a summary of the training history.
