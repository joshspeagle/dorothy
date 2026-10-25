# Preprocessing

This page explains how DOROTHY turns a spectrum and its labels into the
arrays that the model reads. It also defines {term}`normalized units`, the
units of every prediction.

## Spectra

A spectrum enters the model as an array of shape `(3, n_wavelengths)` on its
survey's wavelength grid. A batch of N spectra has shape
`(N, 3, n_wavelengths)`:

| Channel | Content |
|---------|---------|
| 0 | flux f |
| 1 | ρ, the [precision encoding](#precision-encoding) of the flux uncertainty σ = 1/√ivar |
| 2 | mask: 1 at a valid pixel, 0 at an invalid one |

A pixel is valid when its flux is finite and its inverse variance ({term}`ivar`)
is finite and positive.
{py:func}`~dorothy.data.fits_loader.zero_invalid_pixels` sets the flux and
ivar of every other pixel to 0. An invalid pixel therefore enters the model as
0 in all three channels, whatever the catalogue or the FITS file stores there.

::::{container} d-figure
```{raw} html
:file: ../figures/spectrum-input.svg
:class: wide-only
```

```{raw} html
:file: ../figures/spectrum-input-narrow.svg
:class: narrow-only
```

:::{container} caption
The model input: flux, ρ and mask on the survey's wavelength grid, with an
invalid pixel 0 in all three channels; the spectrum is illustrative.
:::
::::

LAMOST MRS has a blue and a red arm on separate wavelength grids. Its input is
the blue arm followed by the red arm, each trimmed to its own grid.

## Flux normalization

The flux scale of a spectrum depends on the distance of the star and on the
calibration of the observation. Neither says anything about the labels.
{py:func}`~dorothy.data.fits_loader.normalize_spectra` therefore scales each
spectrum by the median m and the interquartile range q of its valid pixels:

$$
f \rightarrow \frac{f - m}{q}, \qquad
\mathrm{ivar} \rightarrow q^2\,\mathrm{ivar}, \qquad
\sigma \rightarrow \frac{\sigma}{q} .
$$

The uncertainty σ scales with the flux, so it stays in the units of the flux.
A spectrum without valid pixels keeps m = 0 and q = 1, and a spectrum with
q = 0 keeps q = 1. A pixel whose scaled flux or ivar overflows to infinity
becomes invalid. The checkpoint records whether the model was trained with
this normalization, and prediction repeats it.

## Precision encoding

As a network input, σ has two flaws. A masked pixel has σ = 0, which reads as
a perfect measurement, the opposite of the truth. σ also has no upper bound,
so noisy pixels give large inputs. The second channel therefore holds

$$
\rho = \begin{cases}
\ln(1 + 1/\sigma) & \text{at a valid pixel,} \\
0 & \text{at a masked pixel,}
\end{cases}
$$

where σ is the uncertainty after the flux normalization. ρ grows with the
precision 1/σ. It falls to 0 as σ grows, so a pixel without information
reads like a masked one. For small σ it grows only as ln(1/σ), so the most
precise pixels do not dominate the input.
{py:func}`~dorothy.data.fits_loader.encode_precision` computes ρ after
{py:func}`~dorothy.data.fits_loader.zero_invalid_pixels` and
{py:func}`~dorothy.data.fits_loader.normalize_spectra`, for training,
evaluation and prediction. A checkpoint saved before the encoding reads σ in
this channel ([Old checkpoints](../reference/old-checkpoints.md#precision-encoding)).

```{raw} html
<link rel="stylesheet" href="../_static/explainers/explainer.css">
<div class="d-lab" id="d-precision">
<p class="d-lab-h">Move the slider to change the uncertainty σ of one pixel and read ρ off the curve. Then mask the pixel.</p>
</div>
<script src="../_static/explainers/explainer.js" defer></script>
<script src="../_static/explainers/precision.js" defer></script>
```

## Labels

The labels of N stars have shape `(N, 3, n_parameters)`. The three channels
hold the values y, their uncertainties σ_y, and a mask that is 1 for a valid
label.
Code that reads the mask counts an entry as valid where it is greater than 0.
[Labels and quality flags](../catalogue/labels.md) defines which labels are
valid. The parameters, and their number, come from the catalogue.

## Label normalization

{py:class}`~dorothy.data.normalizer.LabelNormalizer` scales each parameter p
by the median $\tilde y_p$ and the interquartile range $q_p$ of its valid
labels on the training stars:

$$
y'_p = \frac{y_p - \tilde y_p}{q_p}, \qquad \sigma'_p = \frac{\sigma_{y,p}}{q_p} .
$$

For Teff, y is log10 Teff, and its uncertainty is first converted to
σ / (Teff ln 10). One normalized unit of a parameter is therefore the
interquartile range of its training labels. It is in dex for log g and the
abundances, and in dex of log10 Teff for Teff. The model predicts μ and the
predicted scatter s in these units.

::::{container} d-figure
```{raw} html
:file: ../figures/label-normalization.svg
:class: wide-only
```

```{raw} html
:file: ../figures/label-normalization-narrow.svg
:class: narrow-only
```

:::{container} caption
One normalized unit is the interquartile range of the training labels, here
of an illustrative [Fe/H] distribution.
:::
::::

With several label sources, the normalization is fit on all of them. Where two
sources label the same star and parameter, the first source in
`data.label_sources` counts. A parameter without any valid training label gets
$\tilde y_p = 0$ and $q_p = 1$, with a warning.

Masked entries can hold any value, such as a Teff of 0, whose logarithm is
−∞. {py:meth}`~dorothy.data.normalizer.LabelNormalizer.transform_labels`
replaces them with the training median and an uncertainty of 0 before it
transforms, so a masked entry never produces a NaN.
{py:meth}`~dorothy.data.normalizer.LabelNormalizer.inverse_transform` converts
predictions and uncertainties back to kelvin and dex.
