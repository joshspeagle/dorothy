# Preprocessing

## Spectra

A spectrum enters the model as an array of shape `(3, n_wavelengths)`:

| Channel | Content |
|---------|---------|
| 0 | flux |
| 1 | ρ = ln(1 + 1/σ), the [precision encoding](#precision-encoding) of σ = 1/sqrt(ivar) |
| 2 | mask: 1 at valid pixels |

A pixel is valid when its flux is finite and its ivar is finite and positive.
Every loading path and prediction set the flux and ivar of invalid pixels to
0 ({py:func}`~dorothy.data.fits_loader.zero_invalid_pixels`), so an invalid
pixel enters the model as 0 in all three channels, whatever the catalogue or
the FITS file stores there.

```{raw} html
:file: ../figures/spectrum-input.svg
```

The flux scale of a spectrum depends on the star's brightness and on the
calibration of the observation, neither of which says anything about the
labels. With `data.normalize_spectra: true` (the default), each spectrum is
therefore scaled by its own median $m$ and interquartile range $q$ over its
valid pixels:

$$
f \rightarrow \frac{f - m}{q}, \qquad \mathrm{ivar} \rightarrow q^2\,\mathrm{ivar},
$$

so that σ scales with the flux. A spectrum without valid pixels, or with
q = 0, is left unscaled (m = 0, q = 1).
{py:func}`~dorothy.data.fits_loader.normalize_spectra` implements this for
every loading path and for prediction, so a model always sees spectra
preprocessed the same way. The setting is stored in the checkpoint.

LAMOST MRS spectra have a blue and a red arm on separate wavelength grids. The
model input is the blue arm followed by the red arm, each trimmed to its own
grid.

### Precision encoding

As a network input, σ has two flaws. A masked pixel has σ = 0, which reads as
infinite precision, the opposite of the truth. σ also has no upper bound, so
noisy pixels give large inputs. With `data.precision_encoding: true` (the
default), the second channel therefore holds

$$
\rho = \begin{cases}
\ln(1 + 1/\sigma) & \text{at a valid pixel,} \\
0 & \text{at a masked pixel,}
\end{cases}
$$

which grows with the precision 1/σ. ρ falls to 0 as σ grows, so a pixel
without information looks like a masked one, and for small σ it grows only
as ln(1/σ). The encoding comes last, after
{py:func}`~dorothy.data.fits_loader.zero_invalid_pixels` and the
normalization, which rescales σ.
{py:func}`~dorothy.data.fits_loader.encode_precision` implements it for every
loading path and for {py:func}`~dorothy.inference.predictor.predict_from_fits`.
The models read the channel as it comes.

The setting is stored in the checkpoint. A checkpoint saved before this
option has no `precision_encoding` in its configuration, so it loads as
false: its inputs keep σ, and it predicts and evaluates exactly as before.
Training rejects `data.precision_encoding: false`, and `dorothy train --resume`
rejects a checkpoint saved before the encoding.

## Labels

Labels have shape `(3, n_parameters)`: the values, their uncertainties, and a
mask that is 1 for valid labels and 0 otherwise (readers count entries > 0
as valid). A label is valid when its
{term}`flag` is 0 and its uncertainty is positive. With
`data.max_flag_bits` ≥ 0, a star whose flags have more set bits in total than
that value loses all its labels; `-1` keeps every valid label of every star.

{py:class}`~dorothy.data.normalizer.LabelNormalizer` scales each parameter $p$
by its median $\tilde y_p$ and interquartile range $q_p$ over the training
stars:

$$
y'_p = \frac{y_p - \tilde y_p}{q_p}, \qquad \sigma'_p = \frac{\sigma_p}{q_p},
$$

with Teff replaced by log10 Teff before scaling (its uncertainty becomes
σ / (Teff ln 10)). With several label sources, the normalizer is fit on all of
them; where two sources label the same star, the first source in
`data.label_sources` counts. A parameter without any valid training label gets
$\tilde y_p = 0$ and $q_p = 1$, with a warning.

Masked entries can hold any value, including log10(0). Before transforming,
{py:meth}`~dorothy.data.normalizer.LabelNormalizer.transform_labels` replaces
them with the training median and an uncertainty of 0, and the loss selects
valid entries with the mask before any arithmetic, so a masked entry never
produces a NaN.

{py:meth}`~dorothy.data.normalizer.LabelNormalizer.inverse_transform` converts
predictions and uncertainties back to kelvin and dex.
