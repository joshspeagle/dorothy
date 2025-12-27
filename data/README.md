# DOROTHY Data Directory

This directory stores data files for the DOROTHY stellar parameter inference framework.
**This folder is excluded from git** (see `.gitignore`).

## Directory Structure

```
data/
├── README.md              # This file
├── super_catalogue.h5     # Unified multi-survey catalogue (built incrementally)
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

The unified HDF5 super-catalogue (`super_catalogue.h5`) is built incrementally using individual scripts per survey. Each script loads pre-computed training arrays from `data/raw/` and adds them to the HDF5 file.

### Building the Catalogue

Run scripts in order (each adds one survey):

```bash
# 1. BOSS (creates initial HDF5)
python scripts/add_boss_to_hdf5.py

# 2. LAMOST LRS
python scripts/add_lamost_lrs_to_hdf5.py

# 3. LAMOST MRS
python scripts/add_lamost_mrs_to_hdf5.py

# 4. DESI
python scripts/add_desi_to_hdf5.py

# 5. Add Gaia DR3 IDs (replaces 2MASS IDs)
python scripts/add_gaia_ids_to_hdf5.py
```

### Current Status

| Survey | Script | Spectra | Unique Stars | Shape | Status |
|--------|--------|---------|--------------|-------|--------|
| BOSS | `add_boss_to_hdf5.py` | 12,152 | 11,465 | (2, 4506) | ✅ Added |
| LAMOST LRS | `add_lamost_lrs_to_hdf5.py` | 140,765 | 119,355 | (2, 3473) | ✅ Added |
| LAMOST MRS | `add_lamost_mrs_to_hdf5.py` | 71,245 | 58,266 | (4, 3375) | ✅ Added |
| DESI | `add_desi_to_hdf5.py` | 1,421 | 1,308 | (7650,) | ✅ Added |

**Total**: 225,583 spectra from 155,889 unique stars (some overlap between surveys)

### Loading the Catalogue

```python
from dorothy.data import CatalogueLoader

loader = CatalogueLoader("data/super_catalogue.h5")
loader.info()  # Print catalogue summary

# Load BOSS spectra (auto-selects APOGEE labels for BOSS)
data = loader.load(survey="boss")
print(f"Loaded {data.n_stars} stars")

# Get training-ready arrays (2-channel spectra, labels, errors, mask)
X, y, y_err, mask = loader.load_for_training(survey="boss")

# Load multiple surveys (stars observed by both BOSS and LAMOST)
data = loader.load_multi(
    surveys=["boss", "lamost_lrs"],
    mode="intersection"  # Only stars in both surveys
)
print(f"Common stars: {data['boss'].n_stars}")
```

## Pre-computed Training Data

Training arrays are pre-computed on `eridanus.astro.utoronto.ca` and stored in `data/raw/`. These include:
- Normalized spectra (flux + ivar channels)
- APOGEE labels (11 parameters + 11 errors, normalized)
- Cross-match IDs

| Survey | Directory | Files | Spectra |
|--------|-----------|-------|---------|
| BOSS | `data/raw/boss/` | `X_BOSS.npy`, `y_BOSS.npy`, `APOGEE_ids.npy` | 12,152 |
| LAMOST LRS | `data/raw/lamost_lrs/` | `X_lamost.npy`, `y_lamost.npy`, `apogee_ids_lamost.npy` | 140,765 |
| LAMOST MRS | `data/raw/lamost_mrs/` | `X_lamost_mrs2.npy`, `y_lamost_mrs2.npy`, `apogee_ids_lamost_mrs2.npy` | 71,245 |
| DESI | `data/raw/desi/` | `DESI.fits`, `y_BOSS_compare.npy`, `APOGEE_ids.npy` | 1,421 |

## Quality Cuts

### APOGEE Labels (applied during cross-match on server)
- `STARFLAG == 0` (no issues)
- `ASPCAPFLAG == 0` (good ASPCAP solution)
- `SNR > 50` (high signal-to-noise)
- Valid parameters: `Teff > 0`, `logg > -5`, `[Fe/H] > -10`

### Survey-Specific Cuts
- **BOSS**: `ZWARNING == 0`, `SN_MEDIAN > 10`
- **LAMOST LRS**: `snrg > 20` or `snri > 20`
- **LAMOST MRS**: `snr > 20` per arm
- **DESI**: `ZWARN == 0`, `TSNR2_LRG > 0`

## Known Limitations

1. **Duplicate observations**: Some surveys contain multiple spectra per star (repeat observations). Use `CatalogueLoader.load_multi(deduplicate=True)` to keep only one observation per star.
   - BOSS: 687 duplicates (12,152 spectra for 11,465 unique stars)
   - LAMOST LRS: 21,410 duplicates (140,765 spectra for 119,355 unique stars)
   - LAMOST MRS: 12,980 duplicates (71,245 spectra for 58,266 unique stars)
   - DESI: 113 duplicates (1,421 spectra for 1,308 unique stars)
2. **DESI data is limited**: Only ~1,421 stars available locally (vs. full DESI release)
3. **GALAH not yet cross-matched**: GALAH DR4 catalogues are available but not yet integrated
4. **Cross-survey overlap**: Stars observed by multiple surveys can be loaded together using `load_multi(mode="intersection")`
5. **LAMOST MRS batch files overlap**: `X_lamost_mrs.npy` is fully contained in `X_lamost_mrs2.npy`, so only batch 2 is used
6. **One star without Gaia ID**: LAMOST MRS has 1 star that couldn't be matched to Gaia DR3

## Gaia DR3 IDs

The super-catalogue uses Gaia DR3 source IDs (int64) as the primary identifier for cross-matching stars across surveys. These were derived from APOGEE DR17's `GAIAEDR3_SOURCE_ID` column via the 2MASS IDs in the original training files.

Cross-survey overlap (unique stars observed in both surveys):
| Survey 1 | Survey 2 | Common Stars |
|----------|----------|--------------|
| BOSS | LAMOST LRS | 2,062 |
| BOSS | LAMOST MRS | 539 |
| BOSS | DESI | 1,308 |
| LAMOST LRS | LAMOST MRS | 30,915 |
| LAMOST LRS | DESI | 431 |
| LAMOST MRS | DESI | 110 |

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
