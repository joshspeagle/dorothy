# Uncertainties

Every prediction comes with an uncertainty. This page explains what the
uncertainty measures, how the network learns it, and how DOROTHY calibrates
it.

## What the uncertainty measures

The network predicts two numbers for every label: a value μ and a
{term}`predicted scatter` s. The predicted scatter is the expected spread of
the label source's measurements around μ. It is large when the spectrum
constrains the label poorly, or when the label source's own measurements are
noisy. DOROTHY reports the uncertainty c s, where c is the
[calibration factor](#calibration-factors).

::::{container} d-figure
```{raw} html
:file: ../figures/prediction.svg
:class: wide-only
```

```{raw} html
:file: ../figures/prediction-narrow.svg
:class: narrow-only
```

:::{container} caption
A prediction is a distribution: a normal curve centred on μ with width s,
around which the label source measures y; the values are illustrative.
:::
::::

## The loss

Training maximizes the likelihood of each measured label y under a normal
distribution with mean μ and standard deviation s. Up to a constant and a
factor of 2, the loss of one valid label is

$$
\mathcal{L} = \frac{(\mu - y)^2}{s^2} + \ln s^2,
\qquad s^2 = e^{2 \ln s} + s_0^2 .
$$

Here ln s is the model output, clamped to [−10, 5], and $s_0$ is the
{term}`scatter floor`. The first term rewards accurate predictions, weighted by
the predicted variance. The second term stops the network from making s large
to hide its errors. For a given residual μ − y, $\mathcal{L}$ is smallest
where $s^2 = (\mu - y)^2$, so s learns the typical size of the residuals.

:::{note}
The scatter floor $s_0$, in normalized units, stops s from collapsing to
zero. [Configuration](../reference/configuration.md) gives its default.
:::

The loss of a batch is the mean over its valid labels; masked labels add
nothing. {py:func}`~dorothy.losses.heteroscedastic.predicted_scatter` is the
one definition of s, shared by the loss, validation, evaluation and
prediction.

Training starts with a [warm-up](training.md#warm-up) of this loss. `loss: mse`
trains μ alone with a masked mean squared error and leaves ln s untrained.

## The label-error decoder

Every head has two parallel decoders that read the same head input x. The
label decoder $D_\mu$ predicts μ from x alone. The error decoder $D_s$
predicts ln s from x and from u. For each parameter p, u holds the log of the
label source's reported label error $\sigma_{y,p}$ in normalized units:

$$
\mu = D_\mu(x), \qquad
\ln s = D_s(x, u),
$$

$$
u_p = \begin{cases}
\ln \sigma_{y,p} & \text{if } \sigma_{y,p} > 0, \\
m_p & \text{otherwise.}
\end{cases}
$$

Here ln σ_y,p is clamped to [−10, 5] like ln s, and m is the learned
{term}`missing vector`. m starts at 0, a label error of one normalized unit,
far above real label errors.

::::{container} d-figure
```{raw} html
:file: ../figures/uncertainty.svg
:class: wide-only
```

```{raw} html
:file: ../figures/uncertainty-narrow.svg
:class: narrow-only
```

:::{container} caption
The two decoders of a head: μ comes from the head input alone, and ln s also
from the label errors or the missing vector.
:::
::::

No layer that feeds μ reads u. The label errors and the missing labels
correlate with the label values, and prediction has no label errors. In the
multi-head model, x is the trunk output and u enters as a linear map added to
x ([Residual stream](model.md#residual-stream)). In the MLP, x is the input of
the last hidden layer.

The error decoder always has a hidden layer, so that ln s can be a nonlinear
function of u. For example, take a true label noise $\sqrt{a}\,\sigma_y$
that adds in quadrature to the model error $\varepsilon$:

$$
s^2 = a \sigma_y^2 + \varepsilon^2, \qquad
\ln s = \tfrac{1}{2} \ln\left(a\, e^{2u} + \varepsilon^2\right) .
$$

ln s then tends to ln ε for small label errors and to u + ½ ln a for large ones.

In training, masked labels always get m. A fraction
`training.label_error_dropout` of the stars also get m for all their
parameters. A generator seeded by `seed` draws them again for every batch and
head.
The model so learns to predict without the label errors. Validation,
evaluation and prediction always use m, so the best epoch is the best one for
the reported uncertainties.

The predicted scatter s is therefore the expected scatter between the
prediction and the label source's measurement of the star, the label noise
included. No term of $s^2$ is fixed by the reported errors, so s follows the
scatter that the labels show. For a label source that overestimates its
errors, s falls below them.

## Calibration

The predicted scatter can be too small or too large for stars the network did
not train on. The [z-scores](evaluation.md#z-scores) z = (μ − y) / s of the
validation stars test it: with right uncertainties, their robust spread
(P84 − P16) / 2 is 1. P16 and P84 are the 16th and 84th percentiles.

```{raw} html
<link rel="stylesheet" href="../_static/explainers/explainer.css">
<div class="d-lab" id="d-calibration">
<p class="d-lab-h">Drag the slider to scale the predicted scatter of 2,000 simulated validation stars. Watch the z-score histogram match the standard normal curve.</p>
</div>
<script src="../_static/explainers/calibration.js" defer></script>
```

### Calibration factors

After training, the trainer fits one factor c per head and parameter on the
validation stars, the stars that also chose the best epoch:

$$
c = \frac{P_{84} - P_{16}}{2} \ \text{of} \ \frac{\mu - y}{s} ,
\qquad
\sigma_\mathrm{cal} = c\, s .
$$

Here μ and s are the predictions of the best epoch's weights with the
missing vector, in normalized units. c is the validation `z_robust_scatter`,
so the calibrated uncertainty $\sigma_\mathrm{cal}$ gives the validation
z-scores a robust spread of 1. A factor that is not finite and positive is set
to 1. This happens for a parameter with fewer than two validation labels, such
as an element that a label source does not measure.

The checkpoint stores the factors in `calibration.json`, as label source →
parameter → factor. They describe `best_model.pth`.
{py:class}`~dorothy.inference.predictor.Predictor`, `dorothy predict` and
`dorothy evaluate` multiply the predicted scatter by them
({py:func}`~dorothy.inference.calibration.calibrate_scatter`).
`dorothy evaluate` then reports every metric and plot with $\sigma_\mathrm{cal}$,
and lists the factors of each head with the raw and the calibrated
`z_robust_scatter`. `final_model.pth` and the epoch checkpoints have no
factors, and models trained with `loss: mse` have no scatter to calibrate;
these report the raw scatter s. [Commands](../reference/commands.md) lists the
option that turns the factors off.

## What one factor cannot fix

:::{admonition} Limit
:class: caution
One factor per head and parameter rescales the uncertainty of every star by
the same amount. It cannot fix an uncertainty that is too small for faint
stars and too large for bright ones.
:::

The factor also leaves a bias, a median z-score away from 0, in place. It
assumes that new stars resemble the validation stars.
