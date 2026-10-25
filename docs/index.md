---
layout: home
---

# DOROTHY

::::{container} hero
```{image} _static/dorothy.svg
:alt: Dorothy the dinosaur: a teal sauropod with gold star spots, standing on a stellar spectrum
:width: 240px
```

:::{container}
**Stellar labels from low-resolution spectra.**

DOROTHY predicts the temperature, surface gravity and chemical abundances of
a star from its DESI, BOSS or LAMOST spectrum. It learns from stars that
APOGEE or GALAH also observed, and it reports an uncertainty for every
{term}`label`.

[Install](start/install.md) · [Quick start](start/quickstart.md) ·
[What DOROTHY does](start/what-it-does.md)
:::
::::

::::{container} d-figure hero-figure
```{raw} html
:file: figures/label-transfer.svg
:class: wide-only
```

```{raw} html
:file: figures/label-transfer-narrow.svg
:class: narrow-only
```

:::{container} caption
Label transfer: a spectrum goes in, and labels with uncertainties come out;
stars and values are illustrative.
:::
::::

## Three steps

1. **Train on stars in common.** Thousands of stars have both a
   low-resolution spectrum and high-resolution labels. A neural network learns
   to predict the labels from the spectrum.
2. **Predict for everyone else.** The trained network predicts labels for the
   millions of stars that only the low-resolution survey observed.
3. **Report how sure it is.** Every label comes with an uncertainty,
   calibrated on stars the network did not train on.

## The data

All the data live in one HDF5 file, the {term}`super-catalogue`, with one row
per star, matched by Gaia DR3 source ID {cite:p}`gaia2023dr3`.

| Name | Gives | Wavelengths | Resolving power |
|------|-------|-------------|-----------------|
| DESI | spectra | 3600 to 9800 Å | 2000 to 5000 |
| BOSS | spectra | 3600 to 10400 Å | 1560 to 2650 |
| LAMOST LRS | spectra | 3700 to 9000 Å | ≈ 1800 |
| LAMOST MRS | spectra | 4950 to 5350 Å, 6300 to 6800 Å | 7500 |
| APOGEE | labels | 1.51 to 1.70 µm | ≈ 22,500 |
| GALAH | labels | four bands, 4713 to 7887 Å | ≈ 28,000 |

The surveys are described in {cite:t}`desi2022instrument,desi2025dr1` (DESI),
{cite:t}`dawson2013boss,smee2013boss` (BOSS),
{cite:t}`cui2012lamost,zhao2012lamost,liu2020lamostmrs` (LAMOST),
{cite:t}`majewski2017apogee,abdurrouf2022dr17` (APOGEE) and
{cite:t}`buder2021galahdr3,buder2025galahdr4` (GALAH).

## Where to go next

- **[Quick start](start/quickstart.md)**: build a small synthetic catalogue,
  then train, evaluate and use a model on it.
- **[The catalogue](catalogue/contents.md)**: the spectra and labels of every
  star, in one HDF5 file.
- **[Train a model](guides/train.md)**: write a configuration for your
  surveys and label sources, and train on it.
- **[How it works](methods/uncertainties.md)**: how the network learns each
  label and its uncertainty.

```{toctree}
:caption: Start
:hidden:

start/what-it-does
start/install
start/quickstart
```

```{toctree}
:caption: The catalogue
:hidden:

catalogue/contents
catalogue/labels
catalogue/build
catalogue/python
```

```{toctree}
:caption: Guides
:hidden:

guides/predict
guides/read-predictions
guides/train
guides/combine
guides/evaluate
guides/inspect
guides/compare-labels
guides/python
```

```{toctree}
:caption: How it works
:hidden:

methods/overview
methods/preprocessing
methods/model
methods/uncertainties
methods/training
methods/evaluation
```

```{toctree}
:caption: Reference
:hidden:

reference/commands
reference/configuration
reference/file-formats
reference/old-checkpoints
reference/surveys
reference/glossary
reference/api
reference/changelog
reference/bibliography
```

```{toctree}
:caption: Contribute
:hidden:

contribute/code-map
contribute/shared-definitions
contribute/add-a-survey
contribute/writing
contribute/build-docs
```
