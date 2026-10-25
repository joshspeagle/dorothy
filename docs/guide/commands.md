# Commands

`pip install -e .` installs the `dorothy` command. `dorothy <command> --help`
lists every option.

## train

```bash
dorothy train config.yaml
dorothy train config.yaml --device cuda --output-dir runs/
dorothy train config.yaml --resume outputs/old_run/old_run_final
```

Trains the model a configuration describes ([Configuration](configuration.md)),
fits its [calibration factors](../methods/evaluation.md#calibration-factors)
and writes the checkpoint directory `<output_dir>/<name>/<name>_final/`
([Checkpoints](../methods/training.md#checkpoints)).

| Option | Effect |
|--------|--------|
| `--device` | `auto` (default), `cuda`, `mps` or `cpu` |
| `--output-dir` | Overrides `output_dir` |
| `--resume` | Starts from the weights of an earlier checkpoint, with a new optimizer, schedule and warm-up |

## evaluate

```bash
dorothy evaluate <checkpoint>
dorothy evaluate <checkpoint> --format json --output results.json
dorothy evaluate <checkpoint> --model final_model.pth --no-plots
```

Reloads the catalogue with the checkpoint's configuration, takes the test
stars from `data_split.pkl` (or recreates the split from the configured seed
if the file is missing) and scores every output head in normalized and
physical units ([Evaluation](../methods/evaluation.md)). It writes four plots
per label source to `<checkpoint>/plots/evaluation/`: predictions against
labels, residuals, histograms of predictions and labels, and z-score
histograms. The JSON output holds the first head's metrics at the top level
and every head's metrics under `label_sources`.

With the checkpoint's calibration factors, the metrics and plots use the
calibrated uncertainties, and a table per head lists the factors with the raw
and calibrated `z_robust_scatter`
([Calibration factors](../methods/evaluation.md#calibration-factors)). The
JSON output then adds `calibration_factors` and the raw metrics
`uncalibrated_normalized_space` to each head.

| Option | Effect |
|--------|--------|
| `--model` | Weights file to load (default `best_model.pth`) |
| `--format` | `text` (default), `markdown` or `json` |
| `--output` | Writes the results to a file |
| `--plots-dir` | Directory for the plots |
| `--no-plots` | Skips the plots |
| `--no-calibration` | Evaluates the raw predicted scatter, without the calibration factors |
| `--device`, `--batch-size` | Inference device and batch size |

## predict

```bash
dorothy predict --checkpoint <checkpoint> --input spectra.fits \
    --output predictions.csv --survey boss --label-source galah
```

Predicts labels for the spectra in a FITS file and writes one CSV row per
spectrum, with each parameter and its uncertainty (`teff`, `teff_err`, ...)
in physical units. The uncertainty is the predicted scatter, times the
checkpoint's calibration factor for that head and parameter when it has one;
the `calibrated` column says which. The spectra are preprocessed as in
training, including the per-spectrum normalization when the model used it.

The FITS file needs `SPECTRA` (column `FLUX`) and `SPEC_IVAR` (column `IVAR`)
table extensions with one row per spectrum, on the model's wavelength grid.
An optional `WAVELENGTH` extension (column `WAVELENGTH`) holds the grid, and
an `APOGEE` or `GAIA_ID` extension the star IDs. Files with the two LAMOST MRS
arms are not supported yet.

| Option | Effect |
|--------|--------|
| `--checkpoint`, `--input`, `--output` | Checkpoint directory, input FITS file and output CSV (default `predictions.csv`) |
| `--survey` | Survey of the spectra; required for a multi-survey model |
| `--label-source` | Output head to use (default: the first label source) |
| `--normalized` | Writes normalized units instead of physical units |
| `--no-calibration` | Writes the raw predicted scatter, without the calibration factors |
| `--device`, `--batch-size` | Inference device and batch size |

## info

```bash
dorothy info <checkpoint>
```

Prints the model architecture, the uncertainty model (label-aware, or the
additive model of checkpoints saved before it), the configuration, the
parameter names, the label normalization, a summary of the training history
and a table of the calibration factors per parameter and head, or that the
checkpoint has none.
