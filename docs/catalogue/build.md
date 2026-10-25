# How the catalogue is built

`scripts/build_catalogue_v2.py` builds the super-catalogue from cross-matched
FITS files. This page lists the input files, gives the steps to build the
file, and explains how the builder chooses between repeated entries of a star.

## Input files

Each input file pairs one survey with one label source. Its rows are the stars
that both observed, after the cuts in [Upstream selection](#upstream-selection).

| Survey | APOGEE file | GALAH file |
|--------|-------------|------------|
| BOSS | `BOSSxAPOGEE_training.fits` | `BOSSxGALAH_training.fits` |
| DESI | `DESIxAPOGEE_training_dr1.fits` | `DESIxGALAH_training_dr1.fits` |
| LAMOST LRS | `LAMOST_LRSxAPOGEE_training.fits` | `LAMOST_LRSxGALAH_training.fits` |
| LAMOST MRS | `LAMOST_MRSxAPOGEE.fits` | `LAMOST_MRSxGALAH.fits` |

If a DESI file with `_dr1` is missing, the builder reads the same name without
`_dr1`. Each file holds these table extensions, with one row per spectrum:

| Extension | Columns |
|-----------|---------|
| `GAIA_ID` | the Gaia DR3 source ID, in `GAIAEDR3_SOURCE_ID`, `gaiadr3_source_id` or `GAIA_ID` |
| `SPECTRA`, `SPEC_IVAR` | `FLUX` and `IVAR` |
| `WAVELENGTH` | `WAVELENGTH`, in Å |
| `SPECTRA_B`, `SPEC_IVAR_B`, `SPECTRA_R`, `SPEC_IVAR_R` | `FLUX_B`, `IVAR_B`, `FLUX_R`, `IVAR_R`, for LAMOST MRS in place of `SPECTRA` and `SPEC_IVAR` |
| `WAVELENGTH_B`, `WAVELENGTH_R` | `WAVELENGTH_B`, `WAVELENGTH_R`, for LAMOST MRS in place of `WAVELENGTH` |
| `APOGEE` or `GALAH` | the label source's catalogue columns, such as `TEFF`, `TEFF_ERR` and `ASPCAPFLAG` |
| `RVTAB` (optional) | `RA` or `TARGET_RA`, and `DEC` or `TARGET_DEC`, in degrees |

## Step 1: Place the input files

Put the FITS files in `data/raw_catalogues/`. The builder skips a missing file
with a warning, so a subset of the eight files also works. A survey without
input files has no group in the output.

## Step 2: Check the counts

```bash
python scripts/build_catalogue_v2.py --dry-run
```

This prints the number of unique stars and the number of rows in each input
file. It writes no file.

## Step 3: Build the file

```bash
python scripts/build_catalogue_v2.py
```

This writes `data/super_catalogue.h5`, with the 11 default parameters as
[X/Fe]. A build with the extended set of elements and [X/H] abundances is:

```bash
python scripts/build_catalogue_v2.py \
    --elements extended --abundance-scale x_h
```

This writes `data/super_catalogue_xh_extended.h5`.

## Step 4: Check the file

```python
from dorothy.data import CatalogueLoader

CatalogueLoader("data/super_catalogue.h5").info()
```

This prints the number of stars with data for each survey and label source,
and the parameter names.

## Builder options

| Option | Effect | Default |
|--------|--------|---------|
| `--input-dir` | directory of the input files | `data/raw_catalogues` |
| `--output` | path of the HDF5 file | `data/super_catalogue.h5`, with `_xh` and `_extended` added for those builds |
| `--elements` | `core`: 8 elements, 11 parameters; `extended`: 17 elements, 20 parameters | `core` |
| `--abundance-scale` | `x_fe` stores [X/Fe], `x_h` stores [X/H] | `x_fe` |
| `--x-h`, `--X_H` | the same as `--abundance-scale x_h` | |
| `--dry-run` | print the counts and write no file | |
| `-q`, `--quiet` | print less progress for each file | |

[Labels and quality flags](labels.md) describes the parameters of each choice.

## One row per Gaia ID

The builder reads the Gaia DR3 source IDs of all input files. It drops IDs of
0 or less, and gives each remaining ID one row, sorted by ID. A star then gets
its spectrum from each survey and its labels from each label source, wherever
an input file has them.

RA and Dec come from the `RVTAB` extension of an input file whose spectrum the
builder keeps for the star. A star without them has RA and Dec 0.

## Repeat observations

A star can have several spectra from one survey: repeat observations in one
file, or rows in both the APOGEE and the GALAH file of that survey. The
builder keeps the spectrum with the highest signal-to-noise ratio (S/N):

$$
\mathrm{S/N} = \operatorname{median}_{i \,:\, \mathrm{ivar}_i > 0} |f_i| \sqrt{\mathrm{ivar}_i},
$$

where $f_i$ is the flux and $\mathrm{ivar}_i$ the inverse variance of pixel
$i$. For LAMOST MRS the median runs over the pixels of both arms. On a tie, the
builder keeps the earlier spectrum, with the APOGEE file read before the GALAH
file.

Before this, the builder sets the flux and inverse variance of each pixel to 0
where either is not finite, which marks the pixel as missing. It never stores
a spectrum without a valid pixel. The wavelength grid of each survey comes
from its first input file.

## Repeated labels

A star can have several label rows from one label source: one in the file of
each survey that observed it, or several in one file. The builder keeps the
row with the most valid labels. On a tie, it keeps the earlier row, with files
read in the order BOSS, DESI, LAMOST LRS, LAMOST MRS.

## Upstream selection

The input files are made before the builder, outside this repository. Each
survey and label source was cut on quality and S/N:

| Source | Quality cut | S/N cut |
|--------|-------------|---------|
| APOGEE | `(ASPCAPFLAG & STAR_BAD) == 0` | `SNR > 50` |
| GALAH | `flag_sp == 0` | `snr_px_ccd3 > 30` |
| DESI | `RVS_WARN == 0`, `RR_SPECTYPE == 'STAR'` | `SN_B > 20` |
| BOSS | `CLASS == 'STAR'` | `SN_MEDIAN_ALL > 20` |
| LAMOST LRS | `offsets == 0` | `snrg > 20` |
| LAMOST MRS | `rv_br_flag != 1`, `rv_b_flag != 1`, `offsets == 0` | `snr > 20` |

Each survey's stars were matched to APOGEE and to GALAH within 0.5 arcsec. A
star is kept if it matches either label source. BOSS gives Gaia DR3 source IDs
directly, and DESI through a join with its Gaia table. LAMOST, APOGEE and
GALAH stars get them from a positional match to Gaia DR3.
