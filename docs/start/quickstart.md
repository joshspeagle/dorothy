# Quick start

This page trains a BOSS + DESI model on APOGEE labels and uses it.

## 1. Build the catalogue

Put the cross-matched FITS files in `data/raw_catalogues/`, then run:

```bash
python scripts/build_catalogue_v2.py
```

The builder writes `data/super_catalogue.h5`. The
[catalogue build](../catalogue/build.md) page describes the inputs, the options
and the file layout.

## 2. Train

```bash
dorothy train examples/variant2_multi_survey.yaml
```

The configuration names the surveys (`boss`, `desi`), the label source
(`apogee`), the model and the training settings
([Configuration](../reference/configuration.md)). Training prints the losses per epoch and
writes a checkpoint directory:

```
outputs/variant2_multi_survey/variant2_multi_survey_final/
```

[Training](../methods/training.md#checkpoints) lists the files in it.

## 3. Evaluate

```bash
dorothy evaluate outputs/variant2_multi_survey/variant2_multi_survey_final
```

This prints the metrics of each output head on the test split, in normalized
and physical units, and writes plots to `<checkpoint>/plots/evaluation/`.

## 4. Predict

```bash
dorothy predict \
    --checkpoint outputs/variant2_multi_survey/variant2_multi_survey_final \
    --input boss_spectra.fits --survey boss --output predictions.csv
```

The CSV has one row per spectrum, with each parameter and its uncertainty
(`teff`, `teff_err`, ...) in physical units. The uncertainties include the
[calibration factors](../methods/uncertainties.md#calibration-factors) that
training fit.

## From Python

The same steps, and the objects behind them, are on the
[Use DOROTHY from Python](../guides/python.md) page.
