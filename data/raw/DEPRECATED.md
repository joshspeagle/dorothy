# Deprecated Data Files

The `.npy` files in these subdirectories are **deprecated** and kept for reference only.

## What's Deprecated

The following `.npy` files were pre-processed training arrays created by an older pipeline:

- `boss/X_BOSS.npy`, `y_BOSS.npy`, `APOGEE_ids.npy`
- `lamost_lrs/X_lamost.npy`, `y_lamost.npy`, `apogee_ids_lamost.npy`
- `lamost_mrs/X_lamost_mrs2.npy`, `y_lamost_mrs2.npy`, `apogee_ids_lamost_mrs2.npy`
- `desi/y_BOSS_compare.npy`, `APOGEE_ids.npy`

These files:
- Contained already-normalized labels
- Required separate Gaia ID cross-matching
- Only supported APOGEE labels (no GALAH)

## What's Still Valid

The `.fits` files in these subdirectories are still valid reference data:
- `apogee/allStar-dr17-synspec_rev1.fits` - APOGEE DR17 catalogue
- `galah/galah_dr4_allstar_240705.fits` - GALAH DR4 catalogue
- Survey-specific FITS files for cross-reference

## New Data Pipeline

The new v2 pipeline reads from **pre-cross-matched FITS files** in `data/raw_catalogues/`:

```
data/raw_catalogues/
├── BOSSxAPOGEE_training.fits
├── BOSSxGALAH_training.fits
├── DESIxAPOGEE_training.fits
├── DESIxGALAH_training.fits
├── LAMOST_LRSxAPOGEE_training.fits
├── LAMOST_LRSxGALAH_training.fits
├── LAMOST_MRSxAPOGEE.fits
└── LAMOST_MRSxGALAH.fits
```

These files:
- Contain survey spectra already cross-matched with label catalogues
- Have Gaia DR3 IDs directly included
- Store labels in physical units (normalized at training time)
- Support both APOGEE and GALAH labels

## Building the Catalogue

```bash
# Build the v2 catalogue from raw_catalogues/
python scripts/build_catalogue_v2.py --output data/super_catalogue.h5
```
