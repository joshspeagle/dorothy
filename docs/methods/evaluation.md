# Evaluation

{py:func}`~dorothy.inference.evaluation_utils.evaluate_model` scores a
trained model on a set of stars, usually the test set, for every output head.
It uses the batches, label normalization and predicted scatter of training,
so on the validation stars its metrics equal the ones recorded during
training. Given a checkpoint's calibration factors, it scores the calibrated
uncertainties instead ([Calibration factors](#calibration-factors)).
`dorothy evaluate` calls it on the test set, with the factors
([Commands](../guide/commands.md#evaluate)).

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

The z-score of a prediction divides its residual by its predicted scatter:

$$
z = \frac{\mu - y}{s} .
$$

The predicted scatter already includes the label noise
([Scatter and label errors](training.md#scatter-and-label-errors)).
With calibration factors, the z-scores divide by the calibrated uncertainty
$c\,s$ instead ([Calibration factors](#calibration-factors)).
Checkpoints saved before `label_aware_uncertainty` existed predict the scatter
beyond the reported label error σ_y, so their z-scores divide by
$\sqrt{\sigma_y^2 + s^2}$.

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
Calibration can differ between label sources, because each head predicts the
scatter against its own source's labels.

### Calibration factors

Neural networks often report miscalibrated uncertainties, usually
overconfident ones after some overfitting. After training, the trainer
therefore fits one factor c per head and parameter on the validation stars,
the stars that also chose the best epoch:

$$
c = \frac{P_{84} - P_{16}}{2} \ \text{of} \ \frac{\mu - y}{s} ,
\qquad
\sigma_\mathrm{cal} = c\, s ,
$$

where μ and s are the predictions of the best epoch's weights with the
{term}`missing vector`, in normalized units. c is the validation
`z_robust_scatter`, so the calibrated uncertainty σ_cal gives the validation
z-scores a robust spread of 1. A parameter with fewer than two validation
labels, such as an element that a label source does not measure, keeps the
factor 1.

The checkpoint stores the factors in `calibration.json`, as label source →
parameter → factor. They describe `best_model.pth`:
{py:class}`~dorothy.inference.predictor.Predictor`, `dorothy predict` and
`dorothy evaluate` multiply the predicted scatter by them
({py:func}`~dorothy.inference.calibration.calibrate_scatter`).
`dorothy evaluate` then reports every metric and plot with σ_cal, and lists
for each head the factors with the raw and calibrated `z_robust_scatter` of
the evaluated stars.

One factor per head and parameter corrects an uncertainty scale that is wrong
by the same factor for every star. It cannot correct how s depends on the
star: if s is too small for faint stars and too large for bright ones, the
factor rescales both and keeps that pattern. It does not remove a bias, a
median z-score away from 0, and it assumes that new stars resemble the
validation stars.

Checkpoints saved before calibration existed have no `calibration.json`,
and `final_model.pth` and the epoch checkpoints of `training.save_every` have
no factors. These report the raw scatter s, as do models trained with
`loss: mse`, which have no scatter to calibrate. To turn calibration off:

| Where | Setting |
|-------|---------|
| Training | `training.calibrate_uncertainties: false` saves no factors |
| `dorothy predict` | `--no-calibration` |
| `dorothy evaluate` | `--no-calibration` |
| Python | `Predictor.load(..., calibrate=False)`; `evaluate_model` applies factors only when given `calibration=` |
