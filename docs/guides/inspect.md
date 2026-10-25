# Inspect a model

Both analyzers take one spectrum as a torch tensor of shape
`(3, n_wavelengths)`, preprocessed as in training (for example from
{py:meth}`~dorothy.data.catalogue_loader.CatalogueLoader.load_spectrum` with
the checkpoint's `data.normalize_spectra` and `data.precision_encoding`,
converted with `torch.from_numpy`), and its wavelength grid. For a
multi-survey model, `survey` selects the encoder. The analyzers set the
masked pixels to zero in all three channels before the model sees them, so
masked pixels may hold any value, including NaN. They use the model's
predicted scatter s, without the
[calibration factors](../methods/uncertainties.md#calibration-factors).

## Gradient saliency

{py:class}`~dorothy.analysis.saliency.SaliencyAnalyzer` computes the Jacobian
of the predicted values μ and scatters s with respect to the two input
channels at each pixel {cite:p}`simonyan2014saliency`: the flux f and the
error channel σ, which holds the
[precision encoding](../methods/preprocessing.md#precision-encoding) ρ = ln(1 + 1/σ)
unless the checkpoint predates it. The derivatives "with respect to σ" below
are then derivatives with respect to ρ. The analyzer weights the
derivatives by the Fisher information of the predicted normal distribution,
N(μ_p, s_p²), so the importance of pixel i for parameter p is

$$
I_p(i) = \frac{1}{s_p^2}\left[\left(\frac{\partial \mu_p}{\partial f_i}\right)^2
  + \left(\frac{\partial \mu_p}{\partial \sigma_i}\right)^2
  + 2\left(\frac{\partial s_p}{\partial f_i}\right)^2
  + 2\left(\frac{\partial s_p}{\partial \sigma_i}\right)^2\right].
$$

The scatter is $s^2 = e^{2 \ln s} + s_0^2$, as in
[the loss](../methods/uncertainties.md#the-loss), so its derivative with respect to an input
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

## Ablation saliency

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
[Input masking](../methods/training.md#augmentation) draws ln f uniformly in the same
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
