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
python scripts/build_catalogue_v2.py                     # data/super_catalogue.h5

# [X/H] abundances and the extended 17-element set (20 parameters)
python scripts/build_catalogue_v2.py --elements extended --abundance-scale x_h
                                       # data/super_catalogue_xh_extended.h5

# Dry run (print statistics only)
python scripts/build_catalogue_v2.py --dry-run
```

Every row of a v2 catalogue is one star (Gaia DR3 source ID). When a star
appears more than once in the input files, the builder keeps its
highest-S/N spectrum per survey and, per label source, the label row with the
most valid labels (the earlier file on ties).

### Loading the Catalogue

```python
from dorothy.data import CatalogueLoader

loader = CatalogueLoader("data/super_catalogue.h5")
loader.info()                        # print a summary
names = loader.get_parameter_names() # parameter order of the label arrays

# One survey with one label source (stars with both)
X, y = loader.load_for_training(survey="desi", label_source="apogee",
                                max_flag_bits=0, normalize_spectra=True)
# X: (N, 3, n_wave) = [flux, sigma, mask]; y: (N, 3, n_params) = [values, errors, mask]

# Several surveys and label sources, storing spectra only where they exist
data = loader.load_merged_sparse(["boss", "desi", "lamost_mrs"],
                                 label_sources=["apogee", "galah"],
                                 max_flag_bits=-1, normalize_spectra=True)

# One star's spectrum as a model input
X_star, wavelength, gaia_id = loader.load_spectrum("lamost_mrs", 123)
```

Training code does not call these directly: `load_training_data(config)` in
`dorothy/data/training_data.py` picks the right loader for a config and
applies the same filtering for training, evaluation and prediction.

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
│       ├── spectra                        # (N, 4, max_wave) float32 - [flux_b, ivar_b, flux_r, ivar_r]
│       ├── wavelength_b                   # (n_b,) float32 - Blue arm wavelengths
│       ├── wavelength_r                   # (n_r,) float32 - Red arm wavelengths
│       └── snr                            # (N,) float32 - median S/N over both arms
│       # Arms are padded to max_wave = max(n_b, n_r). Loaders trim each arm to
│       # its wavelength grid and concatenate them (blue, then red).
│
├── labels/                                # All arrays have N_TOTAL rows
│   ├── apogee/
│   │   ├── values                         # (N, P) float32 - zeros if missing
│   │   ├── errors                         # (N, P) float32 - zeros if missing or unmeasured
│   │   ├── flags                          # (N, P) uint8 - per-parameter quality
│   │   └── gaia_id                        # (N,) int64 - same as metadata/gaia_id
│   └── galah/
│       └── values, errors, flags, gaia_id
│
└── attrs (HDF5 attributes)
    ├── n_stars: int
    ├── parameter_names: list[str]         # defines P and the label column order
    ├── abundance_scale: "x_fe" or "x_h"
    ├── survey_names: list[str]
    ├── label_sources: list[str]
    ├── creation_date: str
    └── version: str
```

### Deriving Masks

```python
data = loader.load(survey="desi", label_source="apogee")

# Check which stars have data from a survey
has_desi = np.any(data.ivar > 0, axis=1)

# Check which stars have labels
has_apogee = np.any(data.label_errors > 0, axis=1)

# These are computed automatically as data.has_spectrum and data.has_labels
```

## Stellar Parameters

The parameter set is chosen when building (`--elements`, `--abundance-scale`)
and recorded in the `parameter_names` attribute; APOGEE and GALAH use the same
order. The default build stores 11 parameters:

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

`--elements extended` adds O, Na, K, Ti, V, Cr, Co, Ba and Sr (20 parameters),
and `--abundance-scale x_h` stores [X/H] = [X/Fe] + [Fe/H] (errors added in
quadrature; APOGEE's own X_H values where available) instead of [X/Fe].
Elements a label source does not measure are stored with error 0.

## Quality Flags

Flags are stored as binary values (0 = valid, 1 = flagged) derived from source bitmasks. The `flags` array has shape `(N, P)` with one flag per parameter per star. The loaders treat a label as valid when its flag is 0 and its error is positive; with `max_flag_bits >= 0`, a star with more flagged parameters than that has all its labels masked (`-1` keeps every unflagged label).

### APOGEE Flag Criteria

| Parameter | Source | Criteria |
|-----------|--------|----------|
| teff | ASPCAPFLAG | Bits 0 (TEFF_WARN) or 16 (TEFF_BAD) set |
| logg | ASPCAPFLAG | Bits 1 (LOGG_WARN) or 17 (LOGG_BAD) set |
| fe_h | FE_H_FLAG | Any bit set (nonzero) |
| mg_fe | MG_FE_FLAG | Any bit set (nonzero) |
| c_fe | C_FE_FLAG | Any bit set (nonzero) |
| si_fe | SI_FE_FLAG | Any bit set (nonzero) |
| ni_fe | NI_FE_FLAG | Any bit set (nonzero) |
| al_fe | AL_FE_FLAG | Any bit set (nonzero) |
| ca_fe | CA_FE_FLAG | Any bit set (nonzero) |
| n_fe | N_FE_FLAG | Any bit set (nonzero) |
| mn_fe | MN_FE_FLAG | Any bit set (nonzero) |

Additionally, flags are set to 1 if values or errors are NaN/inf. With `--abundance-scale x_h`, a derived [X/H] is also flagged when [Fe/H] is flagged.

**Note on TEFF/LOGG**: Only parameter-specific ASPCAPFLAG bits are used. Other bits (VMICRO_WARN, N_M_WARN, etc.) do not affect TEFF/LOGG flags.

Reference: [APOGEE ASPCAPFLAG Bitmask (SDSS DR17)](https://www.sdss4.org/dr17/irspec/apogee-bitmasks/)

### GALAH Flag Criteria

| Parameter | Source | Criteria |
|-----------|--------|----------|
| teff | flag_sp | Any bit set (nonzero) |
| logg | flag_sp | Any bit set (nonzero) |
| fe_h | flag_fe_h | Any bit set (nonzero) |
| mg_fe | flag_mg_fe | Any bit set (nonzero) |
| c_fe | flag_c_fe | Any bit set (nonzero) |
| si_fe | flag_si_fe | Any bit set (nonzero) |
| ni_fe | flag_ni_fe | Any bit set (nonzero) |
| al_fe | flag_al_fe | Any bit set (nonzero) |
| ca_fe | flag_ca_fe | Any bit set (nonzero) |
| n_fe | flag_n_fe | Any bit set (nonzero) |
| mn_fe | flag_mn_fe | Any bit set (nonzero) |

Reference: [GALAH DR3 Flags](https://www.galah-survey.org/dr3/flags/)

### Using Flags

```python
# Get stars with no flags set (highest quality)
clean_mask = np.all(data.label_flags == 0, axis=1)

# Check specific parameter flag
teff_flagged = data.label_flags[:, 0] > 0
logg_flagged = data.label_flags[:, 1] > 0

# Filter to only unflagged labels
unflagged = data.filter_by_flags(max_flag_bits=0)
```

## Quality Cuts Applied

### Pre-Catalogue Filtering

These filters were applied when the cross-matched input FITS files were made
(upstream of `build_catalogue_v2.py`):

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

The input files were cross-matched with a **UNION** strategy, and the builder
takes the union of their Gaia IDs:

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

## Training with the Catalogue

```yaml
name: desi_apogee
data:
  catalogue_path: data/super_catalogue.h5
  surveys: [desi]
  label_sources: [apogee]
  max_flag_bits: 0        # only stars without flagged parameters
```

See `examples/` for multi-survey and multi-label configurations.

## References

- [APOGEE DR17 Bitmasks](https://www.sdss4.org/dr17/irspec/apogee-bitmasks/)
- [APOGEE Using Parameters](https://www.sdss4.org/dr17/irspec/parameters/)
- [GALAH DR4 Flags](https://www.galah-survey.org/dr4/flags/)
- [GALAH DR4 Best Practices](https://www.galah-survey.org/dr4/using_the_data/)
