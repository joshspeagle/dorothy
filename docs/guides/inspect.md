# Inspect a model

This guide makes saliency maps: plots of how much each pixel of one spectrum
moves the model's predictions.

## What you need

- A checkpoint directory written by `dorothy train` ([Train a
  model](train.md)).
- The catalogue the model was trained on, or one spectrum preprocessed as in
  training.
- Matplotlib, for the plots.

## Step 1: Load the model

```python
from dorothy.inference import load_checkpoint

ckpt = load_checkpoint("outputs/combined/combined_final")
```

{py:func}`~dorothy.inference.checkpoint.load_checkpoint` rebuilds the model
and reads the checkpoint's configuration. For a multi-head model the analyzers
use the first output head.

## Step 2: Load a spectrum

```python
import torch

from dorothy.data import CatalogueLoader

data_config = ckpt.config.data
loader = CatalogueLoader(data_config.catalogue_path)
X, wavelength, gaia_id = loader.load_spectrum(
    "boss", 0,
    normalize_spectra=data_config.normalize_spectra,
    precision_encoding=data_config.precision_encoding,
)
X = torch.from_numpy(X)
```

The analyzers take one spectrum as a torch tensor of shape
`(3, n_wavelengths)`, preprocessed as in training, and its wavelength grid.
{py:meth}`~dorothy.data.catalogue_loader.CatalogueLoader.load_spectrum`
returns a NumPy array, so `torch.from_numpy` converts it. The checkpoint's
`data.normalize_spectra` and `data.precision_encoding` choose the
preprocessing, as in training. Pick a catalogue row that has a spectrum from
the survey; a row without one has no valid pixels.

The analyzers set the masked pixels to zero in all three channels before the
model sees them, as training does. They use the predicted scatter s without
the [calibration factors](../methods/uncertainties.md).

## Step 3: Compute the gradient saliency

{py:class}`~dorothy.analysis.saliency.SaliencyAnalyzer` computes the
derivatives of the prediction μ and the predicted scatter s
{cite:p}`simonyan2014saliency`. It takes them with respect to the two input
channels at each pixel: the flux f and the
[precision encoding](../methods/preprocessing.md) ρ. With
`data.precision_encoding: false`, the second channel holds σ in place of ρ.
It weights the
derivatives by the Fisher information of the predicted normal distribution
N(μ_p, s_p²), so the importance of pixel i for parameter p is

$$
I_p(i) = \frac{1}{s_p^2}\left[\left(\frac{\partial \mu_p}{\partial f_i}\right)^2
  + \left(\frac{\partial \mu_p}{\partial \rho_i}\right)^2
  + 2\left(\frac{\partial s_p}{\partial f_i}\right)^2
  + 2\left(\frac{\partial s_p}{\partial \rho_i}\right)^2\right].
$$

A pixel scores high when a change in it would move the prediction far,
compared with the predicted scatter, or change the scatter. The
derivatives of s are 0 where the clamp of ln s to [−10, 5] is active.

```python
from dorothy.analysis import SaliencyAnalyzer, plot_parameter_saliency

analyzer = SaliencyAnalyzer(
    ckpt.model,
    scatter_floor=ckpt.scatter_floor,
    parameter_names=ckpt.parameter_names,
)
result = analyzer.compute_saliency(X, wavelength, survey="boss", gaia_id=gaia_id)
plot_parameter_saliency(result, "teff", output_path="teff_saliency.png")
```

`survey` selects the encoder of a multi-survey model.

## Step 4: Compute the ablation saliency

{py:class}`~dorothy.analysis.saliency.AblationSaliencyAnalyzer` zeroes one
block of pixels at a time, at every block position, and records how the
predictions change {cite:p}`zeiler2014occlusion`. For each block it reports
the change Δμ of the prediction, the change Δs of the scatter, and the
Fisher-weighted change

$$
F_p = \frac{(\Delta\mu_p)^2 + 2\,(\Delta s_p)^2}{s_p^2},
$$

the form of the gradient saliency with finite changes in place of
derivatives; s_p is the scatter for the whole spectrum. A pixel's importance
is the mean over the blocks that cover it.

::::{container} d-figure
```{raw} html
:file: ../figures/ablation.svg
:class: wide-only
```

```{raw} html
:file: ../figures/ablation-narrow.svg
:class: narrow-only
```

:::{container} caption
Ablation saliency: a pixel's importance is the mean Fisher-weighted change F
over the zeroed blocks that cover it; the data are illustrative.
:::
::::

With `use_training_distribution=True`, the map is the mean of the maps for K
block sizes (`n_block_sizes`, 21 by default):

$$
\bar I(i) = \frac{1}{K}\sum_{k=1}^{K} I_{b_k}(i), \qquad
b_k = \lceil N f_k \rceil, \qquad
\ln f_k = \ln f_{\min} + \frac{k - 1}{K - 1}\ln\frac{f_{\max}}{f_{\min}},
$$

where N is the number of pixels, f_min is 1/N by default, and f_max is 0.5 by
default. [Input masking](../methods/training.md) draws ln f uniformly in the
same way, so the map reflects the kinds of gaps the model saw in training. A
size that several f_k round to counts once for each.

```python
from dorothy.analysis import (
    AblationSaliencyAnalyzer,
    plot_ablation_parameter_saliency,
)

analyzer = AblationSaliencyAnalyzer(
    ckpt.model,
    scatter_floor=ckpt.scatter_floor,
    parameter_names=ckpt.parameter_names,
)
result = analyzer.compute_saliency(
    X, wavelength, survey="boss", use_training_distribution=True
)
plot_ablation_parameter_saliency(result, "teff", output_path="teff_ablation.png")
```

Each block size takes N − b + 1 forward passes, in batches of `batch_size`
(64 by default).

## Step 5: Read the maps

Each plot has three panels for one parameter. Each panel shows the flux
above the precision channel, coloured by the panel's quantity at each pixel.
Masked pixels are grey.

- The gradient plot shows ∂μ/∂x, ∂s/∂x and the importance I. A positive
  ∂μ/∂x means that a higher value at that pixel raises the prediction.
- The ablation plot shows Δμ, Δs and the Fisher-weighted change F. A positive
  Δμ means that removing the block raised the prediction, so the block was
  lowering it.
- The importance panels have no sign. They show where the spectrum carries
  the information about the parameter.

`examples/saliency_example.py` and `examples/ablation_saliency_example.py`
load a checkpoint and a catalogue star and draw both kinds of map.

## Next

- [Read the predictions](read-predictions.md): flag spectra unlike the
  training spectra.
- [The model](../methods/model.md): the layers the saliency passes through.
