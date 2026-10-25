# Add a survey

This page gives the steps to add a survey, a new source of input spectra, to
the catalogue and the models. Only the catalogue builder names the surveys;
the package reads any survey that the catalogue holds.

## Step 1: Write the input files

The builder reads one cross-matched FITS file per pair of a survey and a
label source, in `data/raw_catalogues/`. Each file has these binary-table
extensions, with one row per spectrum.

| Extension | Columns | Content |
|-----------|---------|---------|
| `GAIA_ID` | `GAIAEDR3_SOURCE_ID`, `gaiadr3_source_id` or `GAIA_ID` | Gaia DR3 source ID; rows with an ID ≤ 0 are skipped |
| `SPECTRA` | `FLUX` | Flux, one array per row |
| `SPEC_IVAR` | `IVAR` | Inverse variance, same shape |
| `WAVELENGTH` | `WAVELENGTH` | The wavelength grid in Å, one value per row |
| `APOGEE` or `GALAH` | The label columns of that source | The labels of each row |
| `RVTAB` | `RA` and `DEC`, or `TARGET_RA` and `TARGET_DEC` | Optional: the coordinates |

All files of one survey must use the same wavelength grid: the builder reads
the grid from the first file it finds. The selection cuts and the
cross-match happen before the builder
([How the catalogue is built](../catalogue/build.md)).

## Step 2: Add the survey to the builder

Edit `scripts/build_catalogue_v2.py`. Use a lowercase name, such as `weave`;
it becomes the group `surveys/<name>/` and the name in configurations.

1. Add one entry per label source to `INPUT_FILE_CANDIDATES`:
   `("weave", "apogee"): ["WEAVExAPOGEE_training.fits"]`, and the same for
   `galah`.
2. Add the name to the survey lists in `build_catalogue`: the dry-run loop,
   the survey loop of pass 2, the label loop of pass 3 and the
   `survey_names` file attribute.

A survey with two arms, like LAMOST MRS, also needs the `lamost_mrs`
branches of pass 2, which read `SPECTRA_B`, `SPECTRA_R` and their `IVAR` and
`WAVELENGTH` extensions.

## Step 3: Build the catalogue

1. Check that the builder finds the files:

   ```bash
   python scripts/build_catalogue_v2.py --dry-run
   ```

   It prints the number of stars per survey and label source.

2. Build the catalogue:

   ```bash
   python scripts/build_catalogue_v2.py
   ```

   It writes `data/super_catalogue.h5`. Each survey's line in the final
   summary gives its number of stars with spectra.

## Step 4: Train with the survey

Name the survey in the configuration:

```yaml
data:
  catalogue_path: data/super_catalogue.h5
  surveys: [boss, weave]
  label_sources: [apogee]
```

`dorothy train` reads the survey's number of wavelength bins from the
catalogue and gives the model an encoder for it. The loaders, the model, the
training loop and `dorothy evaluate` need no change.

## Step 5: Predict for the survey

Write the spectra in the [FITS input](../reference/file-formats.md#fits-input)
format, on the survey's catalogue grid, and name the survey:

```bash
dorothy predict --checkpoint outputs/run/run_final --input weave.fits \
    --survey weave
```

## Step 6: Update the docs

1. Add the survey's wavelengths, resolution and LSF, with sources, to
   [Survey properties](../reference/surveys.md).
2. Add its input files to the table in
   [How the catalogue is built](../catalogue/build.md).
3. Add it to the data table of the home page, `docs/index.md`.
4. Add a line under "Unreleased" in `CHANGELOG.md`.
