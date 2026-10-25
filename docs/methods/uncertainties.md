# Uncertainties

## The loss

For one valid label with value y, the heteroscedastic loss is the negative
log-likelihood of y under a normal distribution centred on the prediction μ
with standard deviation s, up to a constant and a factor of 2:

$$
\mathcal{L} = \frac{(\mu - y)^2}{s^2} + \ln s^2,
\qquad s^2 = e^{2 \ln s} + s_0^2 .
$$

The first term rewards accurate predictions, weighted by the predicted
variance; the second keeps the model from inflating its scatter s. The scatter
floor s₀ (`training.scatter_floor`, 0.01 in normalized units by default) stops
s from collapsing to zero. The model output ln s is clamped to [−10, 5] before
use. Masked labels contribute nothing.
{py:func}`~dorothy.losses.heteroscedastic.predicted_scatter` is the one
definition of s, shared by the loss, validation, evaluation and prediction.

### Scatter and label errors

Every head is a {py:class}`~dorothy.models.output.LabelAwareOutput` with two
parallel decoders that read the same head input x. The label decoder D_μ
predicts μ from x alone. The error decoder D_s predicts ln s from x and, for
each parameter p, the label source's reported error σ_y,p in normalized units:

$$
\mu = D_\mu(x), \qquad
\ln s = D_s(x, u), \qquad
u_p = \begin{cases}
\ln \sigma_{y,p} & \text{if } \sigma_{y,p} > 0, \\
m_p & \text{otherwise,}
\end{cases}
$$

where ln σ_y,p is clamped to [−10, 5] like ln s, and m is the learned
{term}`missing vector`. m starts at 0, an error of one normalized unit, far
above real label errors. No layer that feeds μ reads u: the label errors and
the missing labels correlate with the label values, and prediction has no
label errors.

```{raw} html
:file: ../figures/uncertainty.svg
```

Each decoder is a stack of hidden layers (linear, normalization, activation
and optional dropout) followed by a linear layer with one output per
parameter. Both decoders have the same hidden layer sizes, and u enters the
first layer of the error decoder:

Multi-head model
: x is the trunk output, and the hidden layers are the `output_hidden`
  layers. With the residual stream (the default), each hidden layer is a
  residual block and u enters as a linear map added to x
  ([Residual stream](model.md#residual-stream)).

MLP
: The decoders take over the last hidden layer: x is the input of that
  layer, and each decoder has one hidden layer of its size. μ therefore
  passes through exactly the layers of `model.hidden_layers`.

The error decoder always has a hidden layer, so that ln s can be a nonlinear
function of u. If the true label noise is $\sqrt{a}\,\sigma_y$ and adds in
quadrature to the model error $\varepsilon$, for example,

$$
s^2 = a \sigma_y^2 + \varepsilon^2, \qquad
\ln s = \tfrac{1}{2} \ln\left(a\, e^{2u} + \varepsilon^2\right),
$$

which tends to ln ε for small label errors and to u + ½ ln a for large ones.

In training, masked labels always get m. A fraction
`training.label_error_dropout` of the stars (0.5 by default), drawn afresh for
every batch and head from a generator seeded by `seed`, also get m for all
their parameters, so the model learns to predict without the errors.
Validation, evaluation and prediction always use m, so the best epoch is the
best one for the reported uncertainties.

The predicted scatter s is therefore the expected scatter between the
prediction and the label source's measurement of the star, the source's label
noise included. The variance s² has no term fixed by the reported errors, so s
follows the scatter that the labels show: for a label source that
overestimates its errors, s falls below them.

Checkpoints saved before `label_aware_uncertainty` existed have the additive
model: one linear layer predicts μ and ln s, the loss variance is σ_y² + s²,
and s is the scatter beyond the reported label error. Their configuration has
no `label_aware_uncertainty`, so it loads as false: they predict and evaluate
exactly as before. They cannot be trained further, because training rejects
`label_aware_uncertainty: false` and `--resume` needs the same layers.

`loss: mse` trains μ alone with a masked mean squared error and leaves ln s
untrained.

### Several label sources

With several label sources, each head has its own loss, and
`training.label_source_weighting` sets how they combine:

`per_label` (default)
: Every valid label in the batch counts equally. A label source with more
  labels in the batch weighs more.

`per_source`
: Each head's mean loss over its valid labels counts equally, whatever the
  number of labels.

`training.label_source_weights` multiplies each source's contribution, for
example `{apogee: 1.0, galah: 0.5}`.

### Warm-up

The gradient $\partial\mathcal{L}/\partial\mu = 2(\mu - y)/s^2$ is small for
a label with a large s, so its residual shrinks slowly and s stays large
{cite:p}`seitzer2022pitfalls`. Training therefore starts with the β-NLL loss
of that paper, which multiplies the loss of each label by $s^{2\beta}$:

$$
\mathcal{L}_\beta = \operatorname{sg}\left(s^{2\beta}\right)
\left[\frac{(\mu - y)^2}{s^2} + \ln s^2\right],
$$

where sg (stop-gradient) passes the value of its argument but no gradient.
For β = 1, $\partial\mathcal{L}_\beta/\partial\mu = 2(\mu - y)$, the gradient
of the squared error, while s still learns from the residuals. For β = 0,
$\mathcal{L}_\beta = \mathcal{L}$.

β falls linearly from 1 to 0 over the first W optimizer steps of the run and
then stays 0:

$$
\beta_t = \max\left(0,\ 1 - \frac{t}{W}\right), \qquad
W = \operatorname{round}(f\, n_\text{epochs}\, n_\text{steps}),
$$

where t counts the optimizer steps from 0, $n_\text{steps}$ is the number of
steps per epoch and f is `training.nll_warmup_fraction` (0.1 by default; 0
turns the warm-up off). `loss: mse` has no warm-up.

The validation loss is always $\mathcal{L}$, so the best epoch, early stopping
and the `reduce_on_plateau` schedule compare all epochs on one objective.
Early stopping and `reduce_on_plateau` count only the epochs that end after the
warm-up. During the warm-up the model trains on $\mathcal{L}_\beta$, so a
plateau of $\mathcal{L}$ does not show convergence. The training losses in the
history are those of $\mathcal{L}_\beta$, so during the warm-up they are not
comparable with the validation losses. The loss plot shades the warm-up; the
epoch in which it ends mixes both losses.

## Calibration

The z-score of a prediction divides its residual by its predicted scatter:

$$
z = \frac{\mu - y}{s} .
$$

The predicted scatter already includes the label noise
([Scatter and label errors](#scatter-and-label-errors)).
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
z-scores a robust spread of 1. A factor that is not finite and positive is
set to 1. This happens for a parameter with fewer than two validation
labels, such as an element that a label source does not measure.

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
