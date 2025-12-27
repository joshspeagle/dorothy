# DOROTHY Data Directory

This directory stores data files for the DOROTHY stellar parameter inference framework.
**This folder is excluded from git** (see `.gitignore`).

## Directory Structure

```
data/
├── README.md              # This file
├── super_catalogue.h5     # Unified multi-survey catalogue (output of build_catalogue.py)
└── raw/                   # Original survey data files
    ├── apogee/            # APOGEE DR17 stellar parameters
    │   └── allStar-dr17-synspec_rev1.fits
    ├── boss/              # BOSS/SDSS optical spectra catalogue
    │   └── spAll-lite-v6_1_3.fits.gz
    ├── desi/              # DESI spectroscopic data
    │   └── DESI.fits
    ├── galah/             # GALAH DR4 stellar parameters
    │   └── galah_dr4_allstar_240705.fits
    ├── lamost_lrs/        # LAMOST Low Resolution Spectrograph
    │   └── lamost_training.fits
    └── lamost_mrs/        # LAMOST Medium Resolution Spectrograph
        ├── lamost_mrs_training.fits
        └── lamost_mrs_training2.fits
```

## Super-Catalogue

The unified HDF5 super-catalogue (`super_catalogue.h5`) is built using:

```bash
python scripts/build_catalogue.py --config scripts/catalogue_config.yaml
```

See [docs/super_catalogue.md](../docs/super_catalogue.md) for details on the HDF5 schema.

### Loading the Catalogue

```python
from dorothy.data import CatalogueLoader

loader = CatalogueLoader("data/super_catalogue.h5")
loader.info()  # Print catalogue summary

# Load DESI spectra with APOGEE labels
data = loader.load(surveys="desi", label_source="apogee")
print(f"Loaded {data.n_stars} stars, {data.n_complete} with both spectra and labels")

# Get training-ready arrays
X, y, y_err, mask = loader.load_for_training(survey="desi", label_source="apogee")
```

## Raw Data Files

| Survey | File | Size | Description |
|--------|------|------|-------------|
| APOGEE | `allStar-dr17-synspec_rev1.fits` | 3.7 GB | DR17 stellar parameters + Gaia IDs |
| BOSS | `spAll-lite-v6_1_3.fits.gz` | 613 MB | SDSS-V optical spectra catalogue |
| DESI | `DESI.fits` | 167 MB | DESI training cube |
| GALAH | `galah_dr4_allstar_240705.fits` | 723 MB | DR4 stellar parameters + Gaia IDs |
| LAMOST LRS | `lamost_training.fits` | 7.7 GB | Low-res optical training data |
| LAMOST MRS | `lamost_mrs_training*.fits` | 13 GB | Medium-res optical training data |

## Data Sources

| Survey | URL |
|--------|-----|
| APOGEE DR17 | https://www.sdss.org/dr17/irspec/ |
| BOSS/SDSS-V | https://www.sdss.org/dr19/ |
| DESI | https://data.desi.lbl.gov/ |
| GALAH DR4 | https://www.galah-survey.org/dr4/ |
| LAMOST DR11 | http://www.lamost.org/dr11/ |

## Notes

- All data files are excluded from git via `.gitignore`
- Large files (>100 MB) should never be committed
- The `d5martin/` folder contains legacy data from original notebooks (reference only)
- Model checkpoints are saved to experiment-specific output directories, not here

## Storage Requirements

| Category | Size |
|----------|------|
| Raw survey data | ~26 GB |
| Super-catalogue | ~5-10 GB (estimated) |
| **Total** | **~35 GB** |
