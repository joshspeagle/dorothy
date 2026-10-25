# Compare APOGEE and GALAH

This guide measures how well APOGEE and GALAH agree on the stars that both
measure. It reads only the catalogue, so you can check the training
{term}`labels <label>` before any model exists.

## What you need

- The super-catalogue, with both label sources.
- Matplotlib, for the plot in Step 4.

## Step 1: Compute the statistics

```python
from dorothy.analysis import compare_label_sources

comparison = compare_label_sources("data/super_catalogue.h5", max_flag_bits=0)
print(comparison.summary())
```

{py:func}`~dorothy.analysis.label_comparison.compare_label_sources` returns a
{py:class}`~dorothy.analysis.label_comparison.LabelComparison`. Its `summary`
method formats a table with one row per parameter, and `print` prints it. A star counts for a parameter when both sources have a
valid label of it ([Labels and quality flags](../catalogue/labels.md)).
`max_flag_bits` works as `data.max_flag_bits` does in training. The function
matches the stars by Gaia ID, so the sources may store their rows in different
orders. `source_a` and `source_b` choose the two sources; the defaults are
APOGEE and GALAH.

## Step 2: Read the table

For one parameter and one shared star, let y_A and y_B be the labels of the
two sources, and σ_A and σ_B their uncertainties. The function uses

$$
d = y_A - y_B, \qquad
\sigma_d = \sqrt{\sigma_A^2 + \sigma_B^2}, \qquad
z = \frac{d}{\sigma_d}.
$$

If both sources report correct, independent uncertainties and agree on
average, d has mean 0 and standard deviation σ_d. If their errors are also
normal, z follows a standard normal distribution. Over the shared stars, the
table gives:

| Field | Definition |
|-------|------------|
| `n_shared` | number of shared stars |
| `median_offset` | median of d |
| `robust_spread` | (P84 − P16) / 2 of d |
| `predicted_spread` | median of σ_d |
| `ratio` | (P84 − P16) / 2 of z |

P16 and P84 are the 16th and 84th percentiles. All fields except `n_shared`
and `ratio` are in physical units: K for Teff and dex for the others. A
parameter without shared stars has `n_shared` 0 and NaN in the other fields.

## Step 3: Read the ratio

- `ratio` close to 1: the differences have the spread that the uncertainties
  predict.
- `ratio` < 1: the differences are smaller than predicted. At least one
  source overestimates its uncertainties, or the errors of the two sources
  are positively correlated.
- `ratio` > 1: the differences are larger than predicted. At least one source
  underestimates its uncertainties, or the sources differ systematically.

The ratio does not say which source is at fault. `median_offset` shows a
constant offset between the sources.

(P84 − P16) / 2 equals the standard deviation only for normal errors. When
the errors have heavier tails than a normal distribution, correct
uncertainties also give `ratio` < 1; lighter tails give `ratio` > 1. For
errors of unit standard deviation, (P84 − P16) / 2 is −ln(0.32) / √2 = 0.81
for a Laplace distribution. It is 0.68 √3 = 1.18 for a uniform distribution.

`ratio` divides each star's difference by its own σ_d. When the
uncertainties vary from star to star, `robust_spread / predicted_spread` can
differ from 1 even for correct uncertainties, and `ratio` does not. When
every star has the same uncertainties, the two are equal.

## Step 4: Plot the differences

```python
from dorothy.analysis import plot_label_differences

fig = plot_label_differences("data/super_catalogue.h5", max_flag_bits=0)
fig.savefig("label_differences.png", dpi=150)
```

{py:func}`~dorothy.analysis.label_comparison.plot_label_differences` plots d
against the mean Teff and the mean [Fe/H] of the two sources, with one row per
parameter. A panel shows the stars that have a valid label of its parameter,
and of Teff or [Fe/H], in both sources. A trend of d with Teff or [Fe/H] is a
systematic difference between the sources. It widens the spread of d and
raises the ratio.

## Next

- [Combine surveys and label sources](combine.md): train on both label
  sources.
- [Evaluate a model](evaluate.md): the same statistics for a model's
  predictions against the labels.
