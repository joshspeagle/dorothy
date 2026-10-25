# Labels and quality flags

Each {term}`label source` stores a value, an error and a quality {term}`flag`
for every parameter of every star. This page lists the parameters and says
when a {term}`label` is valid.

## Parameters

The file's `parameter_names` attribute lists the parameters in the order of
the label columns. APOGEE and GALAH use the same order. A default build has
11 parameters:

| Column | Name | Parameter | Units |
|--------|------|-----------|-------|
| 0 | `teff` | effective temperature | K |
| 1 | `logg` | surface gravity, log g | dex, with g in cm s⁻² |
| 2 | `fe_h` | [Fe/H] | dex |
| 3–10 | `mg_fe`, `c_fe`, `si_fe`, `ni_fe`, `al_fe`, `ca_fe`, `n_fe`, `mn_fe` | [X/Fe] of Mg, C, Si, Ni, Al, Ca, N and Mn | dex |

## The extended set

With the option `--elements extended`, `scripts/build_catalogue_v2.py` adds
nine elements after Mn: O, Na, K, Ti, V, Cr, Co, Ba and Sr. The catalogue
then has 20 parameters.

A label source does not measure every element. If its input file has no
column for an element, the builder stores that element with error 0. These
labels are never valid.

## [X/Fe] and [X/H]

By default the abundances are relative to iron, [X/Fe]. With the option
`--abundance-scale x_h`, the builder stores them relative to hydrogen, [X/H],
and the names end in `_h`, such as `mg_h`. The file's `abundance_scale`
attribute records the choice.

The builder derives [X/H] from [X/Fe] and [Fe/H]:

$$
[\mathrm{X/H}] = [\mathrm{X/Fe}] + [\mathrm{Fe/H}], \qquad
\sigma_{\mathrm{X/H}} = \sqrt{\sigma_{\mathrm{X/Fe}}^2 + \sigma_{\mathrm{Fe/H}}^2},
$$

where σ is the error of each abundance. GALAH abundances are always derived.
For APOGEE, the builder copies [X/H] from the allStar `X_H` column where the
input file has it, and the error from `X_H_ERR`. It derives [X/H] only for an
element without that column. APOGEE measures no Ba or Sr, so its `ba_h` and
`sr_h` labels have error 0 and are never valid.

## When a label is valid

A label is valid when its error is positive and its flag is 0;
[Preprocessing](../methods/preprocessing.md#labels) gives the masking rule.
The builder flags a label whose value or error is not finite, and stores it
with value 0 and error 0.

In a built catalogue each flag is 0 or 1. The loaders take a `max_flag_bits`
argument, which limits the number of flagged labels of a star. A star with
more flagged labels loses all its labels: `load_labels` and
`load_merged_sparse` mask them, and `load_for_training` drops the star. With
0, one flagged label removes the whole star. With −1, the loaders remove only
the flagged labels.

## Quality flags per label source

The builder sets a label's flag from the flags of its label source:

| Parameter | APOGEE flag set when | GALAH flag set when |
|-----------|----------------------|---------------------|
| `teff` | `ASPCAPFLAG` bit 0 (TEFF_WARN) or bit 16 (TEFF_BAD) | `flag_sp` ≠ 0 |
| `logg` | `ASPCAPFLAG` bit 1 (LOGG_WARN) or bit 17 (LOGG_BAD) | `flag_sp` ≠ 0 |
| `fe_h` | `FE_H_FLAG` ≠ 0 | `flag_fe_h` ≠ 0 |
| each [X/Fe] | `X_FE_FLAG` ≠ 0, such as `MG_FE_FLAG` | `flag_x_fe` ≠ 0, such as `flag_mg_fe` |
| each [X/H] copied from `X_H` | `X_FE_FLAG` ≠ 0 | |
| each derived [X/H] | `X_FE_FLAG` ≠ 0 or `FE_H_FLAG` ≠ 0 | `flag_x_fe` ≠ 0 or `flag_fe_h` ≠ 0 |

Other `ASPCAPFLAG` bits, such as VMICRO_WARN, do not flag Teff or log g. The
bits are defined in the
[APOGEE bitmask documentation](https://www.sdss4.org/dr17/irspec/apogee-bitmasks/)
and the [GALAH flag documentation](https://www.galah-survey.org/dr4/flags/).
