# Predict labels for your spectra

This guide takes a FITS file of spectra and a trained model, and writes a
table of labels with uncertainties.

## What you need

- A trained model: a checkpoint directory written by `dorothy train`
  ([Train a model](train.md)).
- Spectra from one survey the model was trained on, in a FITS file, on that
  survey's wavelength grid.

## Step 1: Check the model

```bash
dorothy info outputs/variant2_multi_survey/variant2_multi_survey_final
```

This prints the model's configuration, its parameters and its
{term}`calibration factors <calibration factor>`. For a multi-head model, it
also lists the surveys, with the number of pixels of each, and the output
heads, one per label source. Your spectra must come from one of these
surveys.

## Step 2: Prepare the FITS file

The file needs two table extensions, with one row per spectrum, in the same
order:

| Extension | Column | Content |
|---|---|---|
| `SPECTRA` | `FLUX` | flux on the survey's wavelength grid |
| `SPEC_IVAR` | `IVAR` | inverse variance of the flux |
| `WAVELENGTH` (optional) | `WAVELENGTH` | the wavelength grid, in Å |
| `APOGEE` (optional) | `APOGEE_ID` or `TARGET_ID` | star identifiers, copied to the output |
| `GAIA_ID` (optional) | the first column | star identifiers, used without `APOGEE` |

A pixel counts as missing when its flux is not finite, or when its inverse
variance is not finite and positive. DOROTHY preprocesses the spectra as in
training, so you do not normalize them yourself.

DOROTHY checks that each spectrum has as many pixels as the model expects. It
does not read the wavelengths, so put the spectra on the grid of the
catalogue: {py:meth}`CatalogueLoader.get_survey_wavelengths
<dorothy.data.catalogue_loader.CatalogueLoader.get_survey_wavelengths>`
returns it.

:::{caution}
**A LAMOST MRS spectrum has two arms.** Write the blue arm followed by the
red arm in one `FLUX` row, each trimmed to its own wavelength grid, as the
catalogue loader does.
:::

## Step 3: Predict

```bash
dorothy predict \
    --checkpoint outputs/variant2_multi_survey/variant2_multi_survey_final \
    --input boss_spectra.fits --survey boss --output predictions.csv
```

`--survey` names the survey of the spectra. A model trained on one survey
does not need it. For a model with two label sources, `--label-source galah`
picks the GALAH head; the default is the first label source.
[Commands](../reference/commands.md) lists the other options.

## Step 4: Read the output

The CSV has one row per spectrum.

| Column | Meaning |
|---|---|
| `star_id` | the star identifier from the file, or the row number without one |
| `teff`, `logg`, `fe_h`, ... | predicted label, in K or dex |
| `teff_err`, `logg_err`, ... | uncertainty of that label, in the same units |
| `calibrated` | true when the uncertainties include the calibration factors |

:::{dorothy}
**The uncertainty is not the error bar of a single measurement.** It is the
expected spread between DOROTHY's prediction and what APOGEE or GALAH would
measure for this star. Read [Uncertainties](../methods/uncertainties.md) to
see how it is checked.
:::

## Next

- [Read the predictions](read-predictions.md): when to trust a label.
- [Use DOROTHY from Python](python.md): predict from arrays with `Predictor`.
