# Quick start

This page trains a model, evaluates it and predicts labels with it, on a
small synthetic catalogue that needs no downloads. Run every command from the
root of the repository, after you [install](install.md) DOROTHY.

## Step 1: Write the synthetic inputs

```bash
python examples/synthetic_quickstart.py
```

The script writes three files into `quickstart/`:

| File | Contents |
|------|----------|
| `catalogue.h5` | A synthetic {term}`super-catalogue` of 2000 stars, in the format of the real one |
| `config.yaml` | A small model that reads BOSS and DESI spectra and predicts APOGEE labels |
| `boss_spectra.fits` | The BOSS spectra of the stars that have no APOGEE labels |

The synthetic stars have four labels: `teff`, `logg`, `fe_h` and `mg_h`.
Their spectra are short (40 BOSS pixels, 30 DESI pixels), so training takes
seconds on a CPU.

## Step 2: Train

```bash
dorothy train quickstart/config.yaml
```

The command loads the catalogue, splits the stars into training, validation
and test sets, and trains for 30 epochs. It ends with lines like these; the numbers can
differ slightly with the platform and the PyTorch version:

```
Training complete!
  Best validation loss: -2.327073
  Best epoch: 27

Saving checkpoint...
  Saved to: quickstart/outputs/quickstart/quickstart_final
```

The checkpoint directory holds the model weights, the configuration, the
split, the [calibration factors](../methods/uncertainties.md#calibration-factors)
and training plots ([Checkpoints](../methods/training.md#checkpoints)).

## Step 3: Evaluate

```bash
dorothy evaluate quickstart/outputs/quickstart/quickstart_final
```

The command predicts the labels of the 163 test stars and compares them with
their APOGEE labels. It prints the metrics in normalized and physical units.
This part shows the robust metrics in physical units:

```
Quantile-based metrics (robust):
--------------------------------------------------------------------------------
Parameter             MAE   Med.Offset  Rob.Scatter
--------------------------------------------------------------------------------
teff             145.5398      +0.1431     159.5721
logg               0.1183      +0.0110       0.1391
fe_h               0.0579      -0.0126       0.0635
mg_h               0.0454      +0.0043       0.0501
```

The command also writes plots to `plots/evaluation/` in the checkpoint
directory.
[Evaluation](../methods/evaluation.md) defines each metric.

## Step 4: Predict

```bash
dorothy predict \
    --checkpoint quickstart/outputs/quickstart/quickstart_final \
    --input quickstart/boss_spectra.fits --survey boss \
    --output quickstart/predictions.csv
```

The model reads two surveys, so `--survey` says which survey the spectra come
from. The command writes one row per spectrum: the star's ID, each label, its
uncertainty in physical units, and whether the uncertainties are calibrated.

```bash
head -3 quickstart/predictions.csv
```

```
star_id,teff,teff_err,logg,logg_err,fe_h,fe_h_err,mg_h,mg_h_err,calibrated
4,3851.993,342.9495,2.4258652,0.12952305,0.035916086,0.076806754,-0.34545624,0.09442433,True
11,3899.3157,785.70544,2.7505932,0.13385144,-0.058377042,0.096448146,-0.25349048,0.1793783,True
```

## On the real catalogue

The same commands work on the real super-catalogue.

1. Build the catalogue from the cross-matched FITS files in
   `data/raw_catalogues/` ([How the catalogue is built](../catalogue/build.md)).
   The builder writes `data/super_catalogue.h5`.

   ```bash
   python scripts/build_catalogue_v2.py
   ```

2. Train a BOSS and DESI model on APOGEE labels.

   ```bash
   dorothy train examples/variant2_multi_survey.yaml
   ```

3. Evaluate it on its test stars.

   ```bash
   dorothy evaluate outputs/variant2_multi_survey/variant2_multi_survey_final
   ```

4. Predict labels for a FITS file of BOSS spectra on the catalogue's BOSS
   wavelength grid.

   ```bash
   dorothy predict \
       --checkpoint outputs/variant2_multi_survey/variant2_multi_survey_final \
       --input boss_spectra.fits --survey boss --output predictions.csv
   ```

[Train a model](../guides/train.md) explains the configuration, and
[Commands](../reference/commands.md) lists every option.
