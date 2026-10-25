# Compare APOGEE and GALAH

APOGEE and GALAH measure some of the same stars.
{py:func}`~dorothy.analysis.label_comparison.compare_label_sources` compares the
difference of their {term}`labels <label>` for these stars with the difference
that their reported uncertainties predict. It reads only the catalogue, so it
can check the training targets before any model exists.

For one parameter and one star, let A be the first source (APOGEE by default)
and B the second (GALAH). The labels are y_A and y_B, and their uncertainties
are σ_A and σ_B. The function uses

$$
d = y_A - y_B, \qquad
\sigma_d = \sqrt{\sigma_A^2 + \sigma_B^2}, \qquad
z = \frac{d}{\sigma_d}.
$$

If both sources report correct, independent uncertainties and agree on
average, d has mean 0 and standard deviation σ_d. If their errors are also
normal, z follows a standard normal distribution.

A star is shared for a parameter when both sources have a valid label of it
([Labels](../methods/preprocessing.md#labels)); `max_flag_bits` works as
`data.max_flag_bits` does. The function matches stars by Gaia ID, so the
sources may store their rows in different orders. For each parameter it
returns, over the shared stars:

| Field | Definition |
|-------|------------|
| `n_shared` | number of shared stars |
| `median_offset` | median of d |
| `robust_spread` | (P84 − P16) / 2 of d |
| `predicted_spread` | median of σ_d |
| `ratio` | (P84 − P16) / 2 of z |

(P84 − P16) / 2 is the `robust_scatter` metric
([Evaluation](../methods/evaluation.md#metrics)). All fields are in physical units (K for
Teff, dex for the others) except `n_shared` and `ratio`. A parameter without
shared stars has `n_shared` 0 and NaN in the other fields.

```python
from dorothy.analysis import compare_label_sources, plot_label_differences

comparison = compare_label_sources("data/super_catalogue.h5", max_flag_bits=0)
print(comparison.summary())
fig = plot_label_differences("data/super_catalogue.h5", max_flag_bits=0)
fig.savefig("label_differences.png", dpi=150)
```

{py:func}`~dorothy.analysis.label_comparison.plot_label_differences` plots d
against the mean Teff and the mean [Fe/H] of the two sources, with one row per
parameter. A panel shows the stars that have a valid label of its parameter,
and of Teff or [Fe/H], in both sources.

## Reading the ratio

- `ratio` close to 1: the differences have the spread that the uncertainties
  predict.
- `ratio` < 1: the differences are smaller than predicted. At least one
  source overestimates its uncertainties, or the errors of the two sources are
  positively correlated.
- `ratio` > 1: the differences are larger than predicted. At least one source
  underestimates its uncertainties, or the sources differ systematically.

(P84 − P16) / 2 equals the standard deviation only for normal errors. When
the errors have heavier tails than a normal distribution, correct
uncertainties also give `ratio` < 1; lighter tails give `ratio` > 1. For
errors of unit standard deviation, (P84 − P16) / 2 is −ln(0.32) / √2 = 0.81 for a
Laplace distribution and 0.68 √3 = 1.18 for a uniform distribution.

The ratio does not say which source is at fault. `median_offset` shows a
constant offset between the sources. A difference that changes with Teff or
[Fe/H] widens the spread of d and raises the ratio. The plot shows it as a
trend.

`ratio` divides each star's difference by its own σ_d. When the uncertainties
vary from star to star, the spread of d mixes the stars' spreads, so
`robust_spread / predicted_spread` can differ from 1 even for correct
uncertainties, and `ratio` does not. When every star has the same
uncertainties, the two are equal.
