# Legacy Catalogue Building Scripts

These scripts are **deprecated** and kept for reference only.

## Why Deprecated?

These scripts were part of the v1 catalogue building pipeline that:
- Read pre-processed `.npy` files from `data/raw/` subdirectories
- Built APOGEE-only catalogues (no GALAH support)
- Required separate Gaia ID cross-matching step

## Current Pipeline

The new v2 pipeline uses `scripts/build_catalogue_v2.py` which:
- Reads directly from pre-cross-matched FITS files in `data/raw_catalogues/`
- Supports both APOGEE and GALAH labels
- Has Gaia DR3 IDs already included in source files
- Produces a unified multi-survey, multi-label catalogue

## Usage

```bash
# NEW: Build v2 catalogue
python scripts/build_catalogue_v2.py --output data/super_catalogue.h5

# OLD: These scripts are no longer used
# python scripts/legacy/add_boss_to_hdf5.py  # Deprecated
```

## Script Descriptions

| Script | Description |
|--------|-------------|
| `add_boss_to_hdf5.py` | Added BOSS spectra from `data/raw/boss/*.npy` |
| `add_desi_to_hdf5.py` | Added DESI spectra from `data/raw/desi/` |
| `add_lamost_lrs_to_hdf5.py` | Added LAMOST LRS from `data/raw/lamost_lrs/*.npy` |
| `add_lamost_mrs_to_hdf5.py` | Added LAMOST MRS from `data/raw/lamost_mrs/*.npy` |
| `add_gaia_ids_to_hdf5.py` | Cross-matched 2MASS IDs to Gaia DR3 |
| `fix_wavelengths.py` | Fixed wavelength grid issues in v1 catalogue |

## Related Files

- Old catalogues: `data/archive/super_catalogue_v1.h5`
- Old raw data: `data/raw/` (`.npy` files are deprecated)
- New raw data: `data/raw_catalogues/` (FITS files)
