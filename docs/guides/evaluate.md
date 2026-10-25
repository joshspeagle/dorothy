# Evaluate a model

This guide scores a trained model on its test stars, and shows how to read the
metrics and plots that it writes.

## What you need

- A checkpoint directory written by `dorothy train` ([Train a
  model](train.md)).
- The catalogue the model was trained on, at the `data.catalogue_path` of the
  checkpoint's `config.yaml`.

## Step 1: Run the evaluation

```bash
dorothy evaluate outputs/boss_apogee/boss_apogee_final
```

The command reloads the catalogue and takes the test stars of the training
run from `data_split.pkl`. These stars took no part in training or in the
choice of the best epoch. It predicts their labels with `best_model.pth` and
the calibrated uncertainties, and compares the predictions with the catalogue
labels.

:::{dorothy}
**Run `dorothy evaluate` from the directory you trained in.** The checkpoint
stores `data.catalogue_path` as you wrote it, so a relative path is read from
the current directory.
:::

## Step 2: Read the metrics

The output has three blocks for each label source:

| Block | Units |
|-------|-------|
| `NORMALIZED SPACE METRICS` | {term}`normalized units`, comparable with the validation metrics of training and across parameters |
| `PHYSICAL SPACE METRICS` | K for Teff, dex for the others |
| `CALIBRATION` | the calibration factor of each parameter, with `z_scatter` before and after calibration |

Each metrics block has three tables with one row per parameter. Each metric
answers one question:

| Question | Metrics |
|----------|---------|
| How far are the predictions from the labels? | `Rob.Scatter`, `RMSE`, `MAE` |
| Are the predictions offset? | `Med.Offset`, `Bias` |
| Are the uncertainties right? | `z_scatter` close to 1, `z_median` close to 0 |
| How large are the uncertainties? | `s_p16`, `s_p50`, `s_p84` |

`Rob.Scatter` and `Med.Offset` come from percentiles, so a few outliers do
not change them; `RMSE`, `Bias` and `SD` include the outliers.
[Evaluation](../methods/evaluation.md) defines each metric.

## Step 3: Look at the plots

The command writes four plots per label source to the checkpoint's
`plots/evaluation/` directory:

| File | Shows |
|------|-------|
| `<source>_pred_vs_truth.png` | prediction against catalogue label, coloured by the uncertainty |
| `<source>_residuals.png` | the density of label minus prediction against the label, with running quartiles |
| `<source>_histograms.png` | the distributions of the labels and the predictions |
| `<source>_zscore_histograms.png` | the z-scores against a standard normal distribution |

Look for a trend in the residuals: a model that predicts too low at one end
of a parameter's range and too high at the other. The z-score histograms
should follow the normal curve; wider histograms mean that the uncertainties
are too small.

## Step 4: Save the results

```bash
dorothy evaluate outputs/boss_apogee/boss_apogee_final \
    --format json --output results.json
```

The JSON file holds the metrics of every label source under `label_sources`,
and those of the first label source also at the top level. `--format
markdown` writes the tables as Markdown. [Commands](../reference/commands.md)
lists the other options, such as `--model` for another weights file.

## Next

- [Read the predictions](read-predictions.md): when to trust a label.
- [Inspect a model](inspect.md): which pixels a prediction depends on.
