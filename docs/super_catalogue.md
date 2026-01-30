# DOROTHY Super-Catalogue

This document describes the unified multi-survey super-catalogue system for DOROTHY, which consolidates spectroscopic data from multiple astronomical surveys cross-matched with stellar parameter labels.

## Overview

The super-catalogue is an HDF5 file that provides:

- **Row-matched structure**: Every star has a row in all arrays; missing data is represented as zeros
- **Multi-survey support**: DESI, BOSS, LAMOST LRS, and LAMOST MRS spectroscopic data
- **Multiple label sources**: APOGEE and GALAH stellar parameters with per-parameter quality flags
- **Unified Gaia DR3 IDs**: All sources cross-matched to a common primary key

## Quick Start

### Building the Catalogue

```bash
# Build from pre-cross-matched FITS files in data/raw_catalogues/
python scripts/build_catalogue_v2.py --output data/super_catalogue.h5

# Dry run (print statistics only)
python scripts/build_catalogue_v2.py --dry-run
```

### Loading the Catalogue

```python
from dorothy.data import CatalogueLoader

# Load and inspect
loader = CatalogueLoader("data/super_catalogue.h5")
loader.info()  # Print summary

# Load DESI spectra with APOGEE labels
data = loader.load(surveys="desi", label_source="apogee")
print(f"Total stars: {data.n_stars}")
print(f"Stars with spectra: {data.n_with_spectra}")
print(f"Stars with labels: {data.n_with_labels}")
print(f"Complete (both): {data.n_complete}")

# Filter to only complete, high-quality stars
complete = data.filter_complete()
high_quality = complete.filter_by_flags(max_flag_bits=0)

# Load directly for training
X, y, y_err, mask = loader.load_for_training(
    survey="desi",
    label_source="apogee",
    max_flag_bits=0
)
```

## HDF5 Schema

```
super_catalogue.h5
├── metadata/                              # All stars (N_TOTAL rows)
│   ├── gaia_id                            # (N,) int64 - Primary key
│   ├── ra                                 # (N,) float64 - From spectroscopic survey
│   └── dec                                # (N,) float64 - From spectroscopic survey
│
├── surveys/                               # All arrays have N_TOTAL rows
│   ├── desi/
│   │   ├── flux                           # (N, 7650) float32 - zeros if missing
│   │   ├── ivar                           # (N, 7650) float32 - zeros if missing
│   │   ├── wavelength                     # (7650,) float32
│   │   └── snr                            # (N,) float32 - 0 if missing
│   ├── boss/
│   │   ├── flux                           # (N, 4506) float32
│   │   ├── ivar                           # (N, 4506) float32
│   │   ├── wavelength                     # (4506,) float32
│   │   └── snr                            # (N,) float32
│   ├── lamost_lrs/
│   │   ├── flux, ivar, wavelength, snr
│   └── lamost_mrs/
│       ├── flux_b, ivar_b, wavelength_b   # Blue arm
│       ├── flux_r, ivar_r, wavelength_r   # Red arm
│       └── snr                            # (N,) float32
│
├── labels/                                # All arrays have N_TOTAL rows
│   ├── apogee/
│   │   ├── values                         # (N, 11) float32 - zeros if missing
│   │   ├── errors                         # (N, 11) float32 - zeros if missing
│   │   └── flags                          # (N, 11) uint8 - per-parameter quality
│   └── galah/
│       ├── values                         # (N, 11) float32 - zeros if missing
│       ├── errors                         # (N, 11) float32 - zeros if missing
│       └── flags                          # (N, 11) uint8 - per-parameter quality
│
└── attrs (HDF5 attributes)
    ├── n_stars: int
    ├── parameter_names: list[str]
    ├── survey_names: list[str]
    ├── creation_date: str
    └── version: str
```

### Deriving Masks

```python
# Check which stars have data from a survey
has_desi = np.any(data.ivar > 0, axis=1)

# Check which stars have labels
has_apogee = np.any(data.label_errors > 0, axis=1)

# These are computed automatically as data.has_spectrum and data.has_labels
```

## Stellar Parameters

The catalogue stores 11 stellar parameters (same order in both APOGEE and GALAH):

| Index | Parameter | Description | Units |
|-------|-----------|-------------|-------|
| 0 | teff | Effective temperature | K |
| 1 | logg | Surface gravity | log(cm/s²) |
| 2 | fe_h | Iron abundance | [Fe/H] dex |
| 3 | mg_fe | Magnesium abundance | [Mg/Fe] dex |
| 4 | c_fe | Carbon abundance | [C/Fe] dex |
| 5 | si_fe | Silicon abundance | [Si/Fe] dex |
| 6 | ni_fe | Nickel abundance | [Ni/Fe] dex |
| 7 | al_fe | Aluminum abundance | [Al/Fe] dex |
| 8 | ca_fe | Calcium abundance | [Ca/Fe] dex |
| 9 | n_fe | Nitrogen abundance | [N/Fe] dex |
| 10 | mn_fe | Manganese abundance | [Mn/Fe] dex |

## Quality Flags

### APOGEE Flags (uint8)

| Bit | Name | Description |
|-----|------|-------------|
| 0 | GRIDEDGE_BAD | PARAMFLAG bit 0 |
| 1 | CALRANGE_BAD | PARAMFLAG bit 1 |
| 2 | OTHER_BAD | PARAMFLAG bit 2 |
| 3 | TEFF_CUT | PARAMFLAG bit 6 |
| 4 | PARAM_SPECIFIC | From ASPCAPFLAG (TEFF_BAD, LOGG_BAD, etc.) |
| 5 | VALUE_BOUNDS | Outside physical bounds |

### GALAH Flags (uint8)

| Bit | Name | Description |
|-----|------|-------------|
| 0 | FLAG_X_FE_NONZERO | flag_X_fe != 0 for this element |
| 1 | VALUE_BOUNDS | Outside physical bounds |

**Important**: GALAH DR4's `flag_fe_h` column is bugged and should not be used. The catalogue builder uses `flag_sp == 0` for pre-filtering.

### Using Flags

```python
# Get stars with no flags set (highest quality)
clean_mask = np.all(data.label_flags == 0, axis=1)

# Check specific flag bit
gridedge_bad = (data.label_flags[:, :] & (1 << 0)) != 0
```

## Quality Cuts Applied

### Pre-Catalogue Filtering

These filters are applied when building the catalogue:

| Source | Filter | SNR Check |
|--------|--------|-----------|
| APOGEE | `(ASPCAPFLAG & STAR_BAD) == 0` | `SNR > 50` |
| GALAH | `flag_sp == 0` | `snr_px_ccd3 > 30` |

### Per-Survey Spectral Filters

| Survey | Filters | SNR Column |
|--------|---------|------------|
| DESI | `RVS_WARN == 0`, `RR_SPECTYPE == 'STAR'` | `SN_B > 20` |
| BOSS | `CLASS == 'STAR'` | `SN_MEDIAN_ALL > 20` |
| LAMOST LRS | `offsets == 0` | `snrg > 20` |
| LAMOST MRS | `rv_br_flag != 1`, `rv_b_flag != 1`, `offsets == 0` | `snr > 20` |

## Cross-Matching Strategy

The catalogue uses a **UNION** cross-matching strategy:

```
For each survey S:
    1. Load & filter survey spectra
    2. Get/cross-match Gaia IDs using spectroscopic RA/DEC
    3. Cross-match to APOGEE (sep < 0.5 arcsec)
    4. Cross-match to GALAH (sep < 0.5 arcsec)
    5. UNION: star included if matched to EITHER label source

Master catalogue = Union of all (Survey × Label) matches
```

### Gaia ID Sources

| Survey | Direct Gaia ID? | Action |
|--------|-----------------|--------|
| BOSS | Yes (`GAIA_ID`) | Use directly |
| DESI | Via join | Join with DESI-Gaia table |
| LAMOST | No | Cross-match with Gaia DR3 |
| APOGEE | No | Cross-match with Gaia DR3 |
| GALAH | No | Cross-match with Gaia DR3 |

## Configuration

See [scripts/catalogue_config.yaml](../scripts/catalogue_config.yaml) for the full configuration reference.

Key options:

```yaml
output:
  path: "data/super_catalogue.h5"
  compression: gzip

gaia:
  dr3_path: "data/raw/gaia/gaia_dr3_sources.fits"
  tolerance_arcsec: 0.5

surveys:
  desi:
    enabled: true
    snr_threshold: 20.0
  # ... more surveys

labels:
  apogee:
    enabled: true
    snr_threshold: 50.0
    apply_star_bad_filter: true
  galah:
    enabled: true
    snr_threshold: 30.0
    apply_flag_sp_filter: true
```

## Training with the Catalogue

```python
from dorothy.config import ExperimentConfig, DataConfig, SurveyType, LabelSource

config = ExperimentConfig(
    name="desi_apogee_training",
    data=DataConfig(
        catalogue_path="data/super_catalogue.h5",
        survey=SurveyType.DESI,
        label_source=LabelSource.APOGEE,
        max_flag_bits=0,  # Only highest quality labels
    ),
)
```

## References

- [APOGEE DR17 Bitmasks](https://www.sdss4.org/dr17/irspec/apogee-bitmasks/)
- [APOGEE Using Parameters](https://www.sdss4.org/dr17/irspec/parameters/)
- [GALAH DR4 Flags](https://www.galah-survey.org/dr4/flags/)
- [GALAH DR4 Best Practices](https://www.galah-survey.org/dr4/using_the_data/)
