# Analysis

`dorothy.analysis` has two tools for understanding a trained model: saliency
maps, which show which wavelengths drive each prediction, and anomaly
detection in the model's latent space. A third tool, the comparison of label
sources, needs only the catalogue.

## Saliency maps

Both analyzers take one spectrum of shape `(3, n_wavelengths)`, preprocessed
as in training (for example from
{py:meth}`~dorothy.data.catalogue_loader.CatalogueLoader.load_spectrum`), and
its wavelength grid. For a multi-survey model, `survey` selects the encoder.
The analyzers set the masked pixels to zero in all three channels before the
model sees them, so masked pixels may hold any value, including NaN. They use
the model's predicted scatter s, without the
[calibration factors](evaluation.md#calibration-factors).

### Gradient saliency

{py:class}`~dorothy.analysis.saliency.SaliencyAnalyzer` computes the Jacobian
of the predicted values μ and scatters s with respect to the flux f and its
uncertainty σ at each pixel {cite:p}`simonyan2014saliency`. It weights the
derivatives by the Fisher information of the predicted normal distribution,
N(μ_p, s_p²), so the importance of pixel i for parameter p is

$$
I_p(i) = \frac{1}{s_p^2}\left[\left(\frac{\partial \mu_p}{\partial f_i}\right)^2
  + \left(\frac{\partial \mu_p}{\partial \sigma_i}\right)^2
  + 2\left(\frac{\partial s_p}{\partial f_i}\right)^2
  + 2\left(\frac{\partial s_p}{\partial \sigma_i}\right)^2\right].
$$

The scatter is $s^2 = e^{2 \ln s} + s_0^2$, as in
[the loss](training.md#the-loss), so its derivative with respect to an input
x is

$$
\frac{\partial s}{\partial x} = \frac{e^{2 \ln s}}{s}\,\frac{\partial \ln s}{\partial x},
$$

and 0 where the clamp of ln s to [−10, 5] is active. A pixel scores high when
changing it would move the prediction by much compared with the predicted
scatter, or would change the scatter itself.

```python
from dorothy.analysis import SaliencyAnalyzer, plot_parameter_saliency

analyzer = SaliencyAnalyzer(model, parameter_names=names, scatter_floor=0.01)
result = analyzer.compute_saliency(X, wavelength, survey="boss")
plot_parameter_saliency(result, "teff", output_path="teff_saliency.png")
```

### Ablation saliency

{py:class}`~dorothy.analysis.saliency.AblationSaliencyAnalyzer` zeroes one
block of pixels at a time, at every block position, and records how the
predictions and their uncertainties change {cite:p}`zeiler2014occlusion`. For
each block it reports the change in μ, the change in s, and the Fisher-weighted
change

$$
F_p = \frac{(\Delta\mu_p)^2 + 2\,(\Delta s_p)^2}{s_p^2},
$$

the quadratic form of gradient saliency with finite changes in place of
derivatives; s_p is the scatter for the unablated spectrum. A pixel's
importance is the average over the blocks that cover it.

```{raw} html
:file: ../figures/ablation.svg
```

With `use_training_distribution=True`, the map is the mean of the maps for K
block sizes (`n_block_sizes`):

$$
\bar I(i) = \frac{1}{K}\sum_{k=1}^{K} I_{b_k}(i), \qquad
b_k = \lceil N f_k \rceil, \qquad
\ln f_k = \ln f_{\min} + \frac{k - 1}{K - 1}\ln\frac{f_{\max}}{f_{\min}},
$$

where N is the number of pixels, f_min defaults to 1/N and f_max to 0.5.
[Input masking](training.md#augmentation) draws ln f uniformly in the same
way, so each b_k has equal weight and the map reflects the kinds of gaps the
model saw in training. A size that several f_k round to counts once for each.

```python
from dorothy.analysis import (
    AblationSaliencyAnalyzer,
    plot_ablation_parameter_saliency,
)

analyzer = AblationSaliencyAnalyzer(model, parameter_names=names)
result = analyzer.compute_saliency(
    X, wavelength, survey="boss", use_training_distribution=True, f_max=0.5
)
plot_ablation_parameter_saliency(result, "teff", output_path="teff_ablation.png")
```

`examples/saliency_example.py` and `examples/ablation_saliency_example.py`
load a checkpoint and a catalogue star and produce both kinds of map.

## Anomaly detection

{py:class}`~dorothy.analysis.knn_anomaly.AnomalyDetector` flags stars whose
embeddings are far from those of the training stars:

1. Compute the embeddings of the training stars and scale them to unit
   length.
2. For each star, find the distance to its k-th nearest training neighbor.
3. Flag the stars whose distance exceeds a percentile of the training
   stars' distances.

```python
from dorothy.analysis import AnomalyDetector

detector = AnomalyDetector.from_predictor(predictor)
detector.fit(X_train, k=10, percentile=99)
result = detector.detect(X_test)
result.get_anomaly_indices()
```

A flagged star has a spectrum unlike those the model was trained on, so its
predictions deserve less trust.

## Comparing label sources

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
([Labels](preprocessing.md#labels)); `max_flag_bits` works as
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
([Evaluation](evaluation.md#metrics)). All fields are in physical units (K for
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

### Reading the ratio

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
