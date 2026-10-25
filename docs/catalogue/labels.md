# Labels and quality flags

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
