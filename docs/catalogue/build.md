# How the catalogue is built

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
