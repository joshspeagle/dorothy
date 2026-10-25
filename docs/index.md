---
layout: home
---

# DOROTHY

::::{container} hero
```{image} _static/dorothy.svg
:alt: Dorothy the dinosaur: a teal sauropod with gold star spots, standing on a stellar spectrum
:width: 240px
```

DOROTHY infers stellar parameters, chemical abundances and their uncertainties
from spectra. It trains neural networks on stars that both a low- or
medium-resolution survey (DESI, BOSS, LAMOST) and a high-resolution survey
(APOGEE, GALAH) have observed. The networks then predict the high-resolution
{term}`labels <label>` from the lower-resolution spectra alone.
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
Label transfer: a spectrum goes in, and labels with uncertainties come out and
place the star in the Kiel diagram; stars and values are illustrative.
:::
::::

```{raw} html
:file: figures/pipeline.svg
```

## The data

| Survey | Role | Wavelengths | Resolution |
|--------|------|-------------|------------|
| DESI | input spectra | 3600–9800 Å | 2000–5000 |
| BOSS | input spectra | 3600–10400 Å | 1560–2650 |
| LAMOST LRS | input spectra | 3700–9000 Å | ≈ 1800 |
| LAMOST MRS | input spectra | 4950–5350 Å and 6300–6800 Å | 7500 |
| APOGEE | labels | 1.51–1.70 µm | ≈ 22,500 |
| GALAH | labels | four optical bands, 4713–7887 Å | ≈ 28,000 |

The surveys are described in {cite:t}`desi2022instrument,desi2025dr1` (DESI),
{cite:t}`dawson2013boss,smee2013boss` (BOSS),
{cite:t}`cui2012lamost,zhao2012lamost,liu2020lamostmrs` (LAMOST),
{cite:t}`majewski2017apogee,abdurrouf2022dr17` (APOGEE) and
{cite:t}`buder2021galahdr3,buder2025galahdr4` (GALAH).

All of it is gathered in one HDF5 file, the {term}`super-catalogue`: one row
per star, identified by its Gaia DR3 source ID {cite:p}`gaia2023dr3`, with the
spectra of every survey that observed the star and the labels of every
{term}`label source` that measured it.

## Where to start

- [What DOROTHY does](start/what-it-does.md) explains the problem and the approach.
- [Quick start](start/quickstart.md) builds a catalogue, trains a model and
  uses it.
- [Code map](contribute/code-map.md) shows where each step lives in the code, and
  the [API reference](reference/api.md) documents every module.

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
contribute/writing
contribute/build-docs
```
