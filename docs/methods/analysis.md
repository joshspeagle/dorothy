# Analysis

`dorothy.analysis` has two tools for understanding a trained model: saliency
maps, which show which wavelengths drive each prediction, and anomaly
detection in the model's latent space.

## Saliency maps

Both analyzers take one spectrum of shape `(3, n_wavelengths)`, preprocessed
as in training (for example from
{py:meth}`~dorothy.data.catalogue_loader.CatalogueLoader.load_spectrum`), and
its wavelength grid. For a multi-survey model, `survey` selects the encoder.
The analyzers set the masked pixels to zero in all three channels before the
model sees them, so masked pixels may hold any value, including NaN.

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
change (Δμ_p)² / s_p² + 2 (Δ ln s_p)². A pixel's importance is the average
over the blocks that cover it.

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
