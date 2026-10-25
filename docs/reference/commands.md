# Commands

This page lists every `dorothy` subcommand with its arguments and defaults.
`pip install -e .` installs the command, and `dorothy <command> --help` prints
the same options.

| Command | Does |
|---------|------|
| [`dorothy train`](#train) | Trains a model from a configuration file and writes a checkpoint |
| [`dorothy evaluate`](#evaluate) | Scores a checkpoint on its test stars |
| [`dorothy predict`](#predict) | Predicts labels for the spectra in a FITS file |
| [`dorothy info`](#info) | Prints what a checkpoint holds |

`dorothy --version` prints the version. `dorothy` without a command prints
the help.

## train

```bash
dorothy train config.yaml
```

| Argument | Default | Effect |
|----------|---------|--------|
| `config` | required | The YAML configuration file ([Configuration](configuration.md)) |
| `--device` | `auto` | `auto`, `cuda`, `mps` or `cpu`. `auto` keeps the configuration's `device` |
| `--output-dir` | the configuration's `output_dir` | Directory for the experiment |
| `--resume` | none | A checkpoint directory. Its `best_model.pth` gives the starting weights; the optimizer, the schedule and the warm-up start new |

The command writes the checkpoint directory
`<output_dir>/<name>/<name>_final/` ([File formats](file-formats.md#checkpoint-directory)).

`--resume` fails when the checkpoint has no `best_model.pth`, was trained
without the precision encoding, has another
`multi_head_model.residual_stream` value, or has weights of another shape.

## evaluate

```bash
dorothy evaluate outputs/run/run_final
```

| Argument | Default | Effect |
|----------|---------|--------|
| `checkpoint` | required | The checkpoint directory |
| `--model` | `best_model.pth` | Weights file in the checkpoint. Only `best_model.pth` has calibration factors |
| `--format` | `text` | `text`, `markdown` or `json` |
| `--output` | none | File for the results, in the `--format` format |
| `--plots-dir` | `<checkpoint>/plots/evaluation` | Directory for the plots |
| `--no-plots` | off | Skips the plots |
| `--no-calibration` | off | Uses the raw predicted scatter, without the calibration factors |
| `--device` | `auto` | `auto`, `cuda`, `mps` or `cpu` |
| `--batch-size` | `1024` | Spectra per forward pass |

The command evaluates the test stars of `data_split.pkl`. Without that file,
it recreates the split from the configured `seed`. It scores every output
head, in normalized and physical units ([Evaluation](../methods/evaluation.md)).

| Output | Content |
|--------|---------|
| Screen | The metrics of each head; with calibration factors, a table of the factors and the raw and calibrated `z_robust_scatter` |
| `--format json` | The first head's metrics at the top level and every head's under `label_sources`. Each head has `normalized_space`, `physical_space` (with a normalizer) and, with calibration factors, `calibration_factors` and `uncalibrated_normalized_space`. Without `--output`, the JSON is the only screen output |
| Plots | Four per label source: `<label_source>_pred_vs_truth.png`, `_residuals.png`, `_histograms.png` and `_zscore_histograms.png` |

## predict

```bash
dorothy predict --checkpoint outputs/run/run_final --input spectra.fits \
    --survey boss --label-source apogee
```

| Argument | Default | Effect |
|----------|---------|--------|
| `--checkpoint` | required | The checkpoint directory |
| `--input` | required | The FITS file of spectra ([FITS input](file-formats.md#fits-input)) |
| `--output` | `predictions.csv` | The CSV file to write ([Predictions CSV](file-formats.md#predictions-csv)) |
| `--survey` | none | Survey of the spectra. Required for a model with several surveys |
| `--label-source` | the first label source | Output head to predict from |
| `--normalized` | off | Writes normalized units instead of physical units |
| `--no-calibration` | off | Writes the raw predicted scatter, without the calibration factors |
| `--device` | `auto` | `auto`, `cuda`, `mps` or `cpu` |
| `--batch-size` | `1024` | Spectra per forward pass |

The spectra get the preprocessing of the checkpoint's configuration
([Preprocessing](../methods/preprocessing.md)). The command fails when the
spectra have another number of wavelength bins than the model's survey.

## info

```bash
dorothy info outputs/run/run_final
```

| Argument | Default | Effect |
|----------|---------|--------|
| `checkpoint` | required | The checkpoint directory |

The command prints these parts of the checkpoint. A file that it cannot read
prints an error line, and the command goes on.

| Part | Content |
|------|---------|
| Model files | Each `.pth` file and its size; the parameter count of `best_model.pth` |
| Configuration | Architecture, uncertainty model (`label-aware` or `additive`), parameter names, `normalize_spectra`, `precision_encoding` |
| Training history | Epochs, final training loss, best validation loss and epoch, per label source |
| Normalizer | The normalized parameters |
| Calibration | The factor per parameter and head, or `none` |
| Learning rates | Number of optimizer steps and the range |

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success, `--help` or `--version` |
| 1 | An error, printed as `Error: …` |
| 2 | Invalid arguments |
