# Evaluation

Evaluation scores a trained model on stars it did not train on. This page
defines the metrics, the z-scores and the two units that they are reported in.

## What evaluation runs

{py:func}`~dorothy.inference.evaluation_utils.evaluate_model` scores a model
on a set of stars for every output head. It uses the batches, the label
normalization and the predicted scatter of training. The model reads no label
errors, so it predicts with its {term}`missing vector`. On the
validation stars, without calibration factors, its metrics therefore equal
the ones recorded during training. Given a checkpoint's
[calibration factors](uncertainties.md#calibration-factors), it scores the
calibrated uncertainty c s instead. `dorothy evaluate` calls it on the test
stars, with the factors ([Commands](../reference/commands.md#evaluate)).

## Metrics

For each parameter, over the stars with a valid label, the metrics describe
the residuals r = μ − y:

| Metric | Definition | Sensitive to outliers |
|--------|------------|-----------------------|
| `rmse` | sqrt(mean(r²)) | yes |
| `bias` | mean(r) | yes |
| `sd` | standard deviation of r | yes |
| `mae` | mean(\|r\|) | less |
| `median_offset` | median(r) | no |
| `robust_scatter` | (P84 − P16) / 2 of r | no |
| `z_median` | median of z | no |
| `z_robust_scatter` | (P84 − P16) / 2 of z | no |
| `pred_unc_p16`, `_p50`, `_p84` | percentiles of the uncertainty, s or c s | no |

P16 and P84 are the 16th and 84th percentiles. For a normal distribution,
(P84 − P16) / 2 equals the standard deviation, and a few outliers change
it little. {py:meth}`~dorothy.inference.evaluator.Evaluator.evaluate_by_survey`
computes the same metrics for the stars of each survey, and the training
history records them on the validation stars.

## Z-scores

The z-score of a prediction divides its residual by its uncertainty:

$$
z = \frac{\mu - y}{c\, s} ,
$$

with c = 1 without calibration factors. The predicted scatter already
includes the label noise
([The label-error decoder](uncertainties.md#the-label-error-decoder)), so
z needs no other term. A checkpoint saved before the label-aware uncertainty
divides by another scatter
([Old checkpoints](../reference/old-checkpoints.md#label-aware-uncertainty)).
If the uncertainties are right, the z-scores of stars the model did not train on follow a standard normal distribution.

::::{container} d-figure
```{raw} html
:file: ../figures/calibration.svg
:class: wide-only
```

```{raw} html
:file: ../figures/calibration-narrow.svg
:class: narrow-only
```

:::{container} caption
The robust spread of the z-scores tells whether the uncertainties are right;
the distributions are illustrative.
:::
::::

| `z_robust_scatter` | Meaning |
|--------------------|---------|
| greater than 1 | The residuals are larger than the uncertainties claim. The model is overconfident. |
| about 1 | The uncertainties are right. |
| less than 1 | The residuals are smaller than the uncertainties claim. The model is underconfident. |

`z_median` measures a bias in units of the uncertainty: 0 for an unbiased
model. `dorothy evaluate` plots a z-score histogram per parameter and label
source. Calibration can differ between label sources, because each head
predicts the scatter against its own source's labels.

## Units

`dorothy evaluate` reports every metric in two units:

- **Normalized units**, the units the model predicts in
  ([Label normalization](preprocessing.md#label-normalization)). These compare
  directly with the training validation metrics and across parameters.
- **Physical units**: kelvin for Teff, dex for log g and the abundances.
  {py:meth}`~dorothy.data.normalizer.LabelNormalizer.inverse_transform`
  converts μ and the uncertainty.

For a parameter normalized linearly, the conversion multiplies the residual
and the uncertainty by the same interquartile range q. Its z-scores are
therefore equal in both units. Teff is normalized in log10, so a normalized
uncertainty s becomes $\mathrm{Teff}\,\ln 10\; q\, s$ kelvin, with Teff the
predicted value. Its z-scores in kelvin equal the normalized ones to first
order in the residual.
