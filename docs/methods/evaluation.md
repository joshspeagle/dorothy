# Evaluation

{py:func}`~dorothy.inference.evaluation_utils.evaluate_model` scores a
trained model on a set of stars, usually the test set, for every output head.
It uses the batches, label normalization and predicted scatter of training,
so on the validation stars its metrics equal the ones recorded during
training, unless training used mixed precision
([Training](training.md#optimization)). `dorothy evaluate` calls it on the
test set ([Commands](../guide/commands.md#evaluate)).

Metrics are reported in two units:

- **Normalized units**, the units the model predicts in. These compare
  directly with the training validation metrics and across parameters.
- **Physical units**: K for Teff, dex for log g and the abundances.

## Metrics

For each parameter, over the stars with a valid label, with residual
r = μ − y:

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
| `pred_unc_p16`, `_p50`, `_p84` | percentiles of the predicted scatter s | no |

P16 and P84 are the 16th and 84th percentiles; for a normal distribution,
(P84 − P16) / 2 equals the standard deviation.
{py:meth}`~dorothy.inference.evaluator.Evaluator.evaluate_by_survey` computes
the same metrics for the stars of each survey.

## Calibration

The z-score of a prediction divides its residual by its total uncertainty:

$$
z = \frac{\mu - y}{\sqrt{\sigma_y^2 + s^2}} .
$$

If the model's uncertainties are right, the z-scores of held-out stars follow
a standard normal distribution, and `z_robust_scatter` is close to 1.

```{raw} html
:file: ../figures/calibration.svg
```

- `z_robust_scatter` > 1: the residuals are larger than the uncertainties
  claim; the model is overconfident.
- `z_robust_scatter` < 1: the uncertainties are larger than the residuals;
  the model is underconfident.

`dorothy evaluate` plots a z-score histogram per parameter and label source.
Calibration can differ between label sources, because each source's reported
uncertainties σ_y enter the z-score.
