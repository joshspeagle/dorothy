# The super-catalogue

The super-catalogue is one HDF5 file with every star's spectra and labels.
Each row is one star, identified by its Gaia DR3 source ID. For each survey,
the star's row holds its spectrum, or zeros if the survey did not observe it;
for each label source, its labels, or zeros if the source did not measure
them.

```{raw} html
:file: ../figures/catalogue.svg
```

## Building it

`scripts/build_catalogue_v2.py` reads the cross-matched FITS files in
`data/raw_catalogues/`, one per pair of a survey and a label source:

| Survey | APOGEE file | GALAH file |
|--------|-------------|------------|
| BOSS | `BOSSxAPOGEE_training.fits` | `BOSSxGALAH_training.fits` |
| DESI | `DESIxAPOGEE_training_dr1.fits` (or without `_dr1`) | `DESIxGALAH_training_dr1.fits` (or without `_dr1`) |
| LAMOST LRS | `LAMOST_LRSxAPOGEE_training.fits` | `LAMOST_LRSxGALAH_training.fits` |
| LAMOST MRS | `LAMOST_MRSxAPOGEE.fits` | `LAMOST_MRSxGALAH.fits` |

```bash
python scripts/build_catalogue_v2.py                  # data/super_catalogue.h5
python scripts/build_catalogue_v2.py --elements extended --abundance-scale x_h
                                                      # data/super_catalogue_xh_extended.h5
python scripts/build_catalogue_v2.py --dry-run        # print the counts only
```

The builder collects the Gaia IDs of all files, gives each star one row, and
fills the rows survey by survey and label source by label source. A star can
appear in several files, or more than once in one file (repeat
observations). For each survey the builder keeps the star's spectrum with the
highest signal-to-noise ratio (S/N, the median of |flux| · sqrt(ivar) over
valid pixels; over both arms for LAMOST MRS). For each label source it keeps
the row with the most valid labels, and the earlier file when two rows tie.
Pixels with non-finite flux or ivar are stored with ivar = 0, which masks them.

## Parameters

The parameters are chosen when building and recorded in the file's
`parameter_names` attribute, in the order of the label columns; APOGEE and
GALAH use the same order. The default build has 11:

| Index | Name | Parameter | Units |
|-------|------|-----------|-------|
| 0 | `teff` | Effective temperature | K |
| 1 | `logg` | Surface gravity | dex (log cm s⁻²) |
| 2 | `fe_h` | [Fe/H] | dex |
| 3–10 | `mg_fe`, `c_fe`, `si_fe`, `ni_fe`, `al_fe`, `ca_fe`, `n_fe`, `mn_fe` | [X/Fe] | dex |

`--elements extended` adds O, Na, K, Ti, V, Cr, Co, Ba and Sr (20 parameters).
`--abundance-scale x_h` stores [X/H] instead of [X/Fe]: APOGEE's own X_H
values where the allStar file has them, otherwise [X/Fe] + [Fe/H] with the
errors added in quadrature. An element a label source does not measure is
stored with error 0, which masks it.

## Layout

```
super_catalogue.h5
├── metadata/
│   ├── gaia_id          (N,) int64, sorted; the row's star
│   ├── ra, dec          (N,) float64, degrees (0 where unknown)
├── surveys/
│   ├── desi/, boss/, lamost_lrs/
│   │   ├── flux         (N, n_wavelengths) float32, 0 where missing
│   │   ├── ivar         (N, n_wavelengths) float32, 0 where missing or masked
│   │   ├── wavelength   (n_wavelengths,) float32, Å
│   │   └── snr          (N,) float32, 0 where missing
│   └── lamost_mrs/
│       ├── spectra      (N, 4, max(n_b, n_r)) float32: flux_b, ivar_b, flux_r, ivar_r
│       ├── wavelength_b (n_b,) float32, Å
│       ├── wavelength_r (n_r,) float32, Å
│       └── snr          (N,) float32
└── labels/
    ├── apogee/, galah/
    │   ├── values       (N, P) float32, 0 where missing
    │   ├── errors       (N, P) float32, 0 where missing or not measured
    │   ├── flags        (N, P) uint8, 0 = no warning
    │   └── gaia_id      (N,) int64, same as metadata/gaia_id
```

The file attributes are `n_stars`, `parameter_names`, `abundance_scale`
(`x_fe` or `x_h`), `survey_names`, `label_sources`, `creation_date` and
`version`. The LAMOST MRS arms are padded to the longer arm; the loaders trim
each arm to its wavelength grid and concatenate them, blue then red.

## Quality flags

A label is valid when its error is positive and its flag is 0. The builder
derives the flags from each catalogue's own flags, and also sets them when a
value or error is not finite:

| Parameter | APOGEE | GALAH |
|-----------|--------|-------|
| `teff` | ASPCAPFLAG bit 0 (TEFF_WARN) or 16 (TEFF_BAD) | `flag_sp` ≠ 0 |
| `logg` | ASPCAPFLAG bit 1 (LOGG_WARN) or 17 (LOGG_BAD) | `flag_sp` ≠ 0 |
| `fe_h` | `FE_H_FLAG` ≠ 0 | `flag_fe_h` ≠ 0 |
| [X/Fe] | `X_FE_FLAG` ≠ 0 | `flag_x_fe` ≠ 0 |

Other ASPCAPFLAG bits (such as VMICRO_WARN) do not flag Teff or log g. A
derived [X/H] is also flagged when [Fe/H] is flagged. References:
[APOGEE bitmasks](https://www.sdss4.org/dr17/irspec/apogee-bitmasks/),
[GALAH flags](https://www.galah-survey.org/dr4/flags/).

## How the input files were made

The cross-matched files come from upstream of the builder, with these cuts:

| Source | Selection | S/N |
|--------|-----------|-----|
| APOGEE | `(ASPCAPFLAG & STAR_BAD) == 0` | `SNR > 50` |
| GALAH | `flag_sp == 0` | `snr_px_ccd3 > 30` |
| DESI | `RVS_WARN == 0`, `RR_SPECTYPE == 'STAR'` | `SN_B > 20` |
| BOSS | `CLASS == 'STAR'` | `SN_MEDIAN_ALL > 20` |
| LAMOST LRS | `offsets == 0` | `snrg > 20` |
| LAMOST MRS | `rv_br_flag != 1`, `rv_b_flag != 1`, `offsets == 0` | `snr > 20` |

Each survey's stars were matched to APOGEE and GALAH within 0.5 arcsec, and a
star is kept if it matches either label source. BOSS provides Gaia IDs
directly; DESI through a join with its Gaia table; LAMOST, APOGEE and GALAH
through a positional match to Gaia DR3.

## Loading

{py:class}`~dorothy.data.catalogue_loader.CatalogueLoader` reads the file.
Training code does not call it directly:
{py:func}`~dorothy.data.training_data.load_training_data` picks the loader a
configuration needs ([Training](training.md#data-selection-and-split)).

```python
from dorothy.data import CatalogueLoader

loader = CatalogueLoader("data/super_catalogue.h5")
loader.info()                          # summary of surveys and label sources
names = loader.get_parameter_names()   # label column order

# One survey and one label source, stars with both, as model inputs
X, y = loader.load_for_training(survey="desi", label_source="apogee",
                                max_flag_bits=0, normalize_spectra=True)

# Several surveys and label sources, spectra stored only where they exist
data = loader.load_merged_sparse(["boss", "desi", "lamost_mrs"],
                                 label_sources=["apogee", "galah"],
                                 max_flag_bits=-1, normalize_spectra=True)

# One star's spectrum as a model input
X_star, wavelength, gaia_id = loader.load_spectrum("lamost_mrs", 123)
```

With several surveys, most stars lack most surveys. The sparse loader
({py:class}`~dorothy.data.catalogue_loader.SparseMergedData`) reads and keeps
each survey's spectra only for the stars that have them, with an index from
catalogue rows to stored rows. The dense alternative
(`data.use_dense_loading: true`) stores full arrays, zeros included, and needs
several times more memory.
