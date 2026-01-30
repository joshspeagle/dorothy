# DOROTHY Data Directory

This directory stores data files for the DOROTHY stellar parameter inference framework.
**This folder is excluded from git** (see `.gitignore`).

## Directory Structure

```
data/
├── README.md                     # This file
├── super_catalogue.h5            # Unified multi-survey, multi-label catalogue (v2)
├── raw_catalogues/               # Pre-cross-matched training FITS files
│   ├── BOSSxAPOGEE_training.fits
│   ├── BOSSxGALAH_training.fits
│   ├── DESIxAPOGEE_training.fits
│   ├── DESIxGALAH_training.fits
│   ├── LAMOST_LRSxAPOGEE_training.fits
│   ├── LAMOST_LRSxGALAH_training.fits
│   ├── LAMOST_MRSxAPOGEE.fits
│   └── LAMOST_MRSxGALAH.fits
├── raw/                          # Reference survey catalogues
│   ├── apogee/                   # APOGEE DR17 stellar parameters
│   ├── boss/                     # BOSS/SDSS optical spectra catalogue
│   ├── desi/                     # DESI spectroscopic data
│   ├── galah/                    # GALAH DR4 stellar parameters
│   ├── lamost_lrs/               # LAMOST LRS spectra
│   ├── lamost_mrs/               # LAMOST MRS spectra
│   └── DEPRECATED.md             # Note about legacy .npy files
└── archive/                      # Archived v1 catalogues
    ├── super_catalogue_v1.h5
    └── super_catalogue_v1_clean.h5
```

## Super-Catalogue (v2)

The unified HDF5 super-catalogue (`super_catalogue.h5`) contains multi-survey spectra with both APOGEE and GALAH stellar parameter labels.

### Building the Catalogue

```bash
# Build from pre-cross-matched FITS files
python scripts/build_catalogue_v2.py --output data/super_catalogue.h5

# Dry run (print statistics only)
python scripts/build_catalogue_v2.py --dry-run
```

### Current Status (v2)

| Survey | APOGEE Stars | GALAH Stars | Wavelengths |
|--------|-------------|-------------|-------------|
| BOSS | 10,693 | 1,370 | 4,506 |
| DESI | 38,732 | 24,106 | 7,650 |
| LAMOST LRS | 140,765 | 57,764 | 3,473 |
| LAMOST MRS | 71,245 | 25,001 | 3,375 (blue) + 3,248 (red) |

**Total**: ~250k unique stars (union across all surveys and label sources)

### Loading the Catalogue

```python
from dorothy.data import CatalogueLoader

loader = CatalogueLoader("data/super_catalogue.h5")
loader.info()  # Print catalogue summary

# Load single survey with APOGEE labels
data = loader.load(survey="boss", label_source="apogee")
print(f"Loaded {data.n_stars} stars")

# Load single survey with GALAH labels
data = loader.load(survey="desi", label_source="galah")

# Multi-survey sparse loading (memory efficient)
sparse = loader.load_merged_sparse(
    surveys=["boss", "desi", "lamost_lrs", "lamost_mrs"],
    label_sources=["apogee"]
)
print(f"Total stars: {sparse.n_total}")

# Multi-survey, multi-label loading
sparse = loader.load_merged_sparse(
    surveys=["boss", "desi"],
    label_sources=["apogee", "galah"]
)
```

## Raw Catalogues (data/raw_catalogues/)

Pre-cross-matched FITS files combining survey spectra with stellar parameter labels.
These files are the source data for building the super-catalogue.

| File | Survey | Labels | Stars | Size |
|------|--------|--------|-------|------|
| BOSSxAPOGEE_training.fits | BOSS | APOGEE | 10,693 | 0.8 GB |
| BOSSxGALAH_training.fits | BOSS | GALAH | 1,370 | 0.1 GB |
| DESIxAPOGEE_training.fits | DESI | APOGEE | 38,732 | 4.6 GB |
| DESIxGALAH_training.fits | DESI | GALAH | 24,106 | 2.8 GB |
| LAMOST_LRSxAPOGEE_training.fits | LAMOST LRS | APOGEE | 140,765 | 8.0 GB |
| LAMOST_LRSxGALAH_training.fits | LAMOST LRS | GALAH | 57,764 | 3.1 GB |
| LAMOST_MRSxAPOGEE.fits | LAMOST MRS | APOGEE | 71,245 | 7.2 GB |
| LAMOST_MRSxGALAH.fits | LAMOST MRS | GALAH | 25,001 | 2.5 GB |

Each FITS file contains:
- `SPECTRA` / `SPECTRA_B`, `SPECTRA_R`: Spectral flux
- `SPEC_IVAR` / `SPEC_IVAR_B`, `SPEC_IVAR_R`: Inverse variance
- `WAVELENGTH` / `WAVELENGTH_B`, `WAVELENGTH_R`: Wavelength grid
- `APOGEE` or `GALAH`: Full label catalogue with all stellar parameters
- `GAIA_ID`: Gaia DR3 source IDs

## Schema Versions

### v2 (Current)
- Unified label groups: `/labels/apogee/`, `/labels/galah/`
- Supports both APOGEE and GALAH labels
- Built from `raw_catalogues/` FITS files
- Labels in physical units (normalized at training time)

### v1 (Archived)
- Per-survey label groups: `/labels/apogee_boss/`, `/labels/apogee_desi/`, etc.
- APOGEE labels only (no GALAH)
- Built from legacy `.npy` files in `raw/`
- Archived in `data/archive/`

## Gaia DR3 IDs

The super-catalogue uses Gaia DR3 source IDs (int64) as the primary identifier for cross-matching stars across surveys and label sources.

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
- Model checkpoints are saved to experiment-specific output directories, not here
- Legacy scripts are archived in `scripts/legacy/`

## Storage Requirements

| Category | Size |
|----------|------|
| Raw catalogues (raw_catalogues/) | ~29 GB |
| Super-catalogue (super_catalogue.h5) | ~15-20 GB |
| Reference data (raw/) | ~26 GB |
| **Total** | **~70 GB** |
