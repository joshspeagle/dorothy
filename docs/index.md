# DOROTHY

DOROTHY infers stellar parameters, chemical abundances and their uncertainties
from spectra. It trains neural networks on stars that both a low- or
medium-resolution survey (DESI, BOSS, LAMOST) and a high-resolution survey
(APOGEE, GALAH) have observed. The networks then predict the high-resolution
{term}`labels <label>` from the lower-resolution spectra alone.

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

- [Concepts](background/concepts.md) explains the problem and the approach.
- [Quick start](guide/quickstart.md) builds a catalogue, trains a model and
  uses it.
- [Code map](guide/code-map.md) shows where each step lives in the code, and
  the [API reference](reference/api.md) documents every module.

```{toctree}
:caption: Background
:hidden:

background/concepts
background/glossary
```

```{toctree}
:caption: Methods
:hidden:

methods/catalogue
methods/preprocessing
methods/models
methods/training
methods/evaluation
methods/analysis
```

```{toctree}
:caption: Using DOROTHY
:hidden:

guide/installation
guide/quickstart
guide/configuration
guide/commands
guide/python-api
guide/code-map
```

```{toctree}
:caption: Reference
:hidden:

reference/api
reference/line-spread-functions
reference/changelog
reference/bibliography
```
