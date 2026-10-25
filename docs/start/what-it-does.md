# What DOROTHY does

DOROTHY predicts the labels of a star from a low-resolution spectrum. This
page explains the idea in four parts, one figure each.

## Labels from spectra

A star's spectrum encodes its atmospheric parameters and its composition. These
are the effective temperature (Teff), the surface gravity (log g), the iron
abundance [Fe/H] and the abundances of other elements. Each of these numbers
is one {term}`label` of the star.

High-resolution surveys measure labels precisely. APOGEE observes the H band
(1.51 to 1.70 µm) at a resolving power of about 22,500
{cite:p}`majewski2017apogee`, and its pipeline fits synthetic spectra to each
observation {cite:p}`garciaperez2016aspcap`. GALAH observes four optical bands
at a resolving power of about 28,000 {cite:p}`buder2021galahdr3`.

Low-resolution surveys observe many more stars. DESI
{cite:p}`desi2022instrument`, BOSS {cite:p}`dawson2013boss` and LAMOST
{cite:p}`cui2012lamost` have resolving powers between about 1500 and 7500. At
these resolutions, neighbouring absorption lines blend, so a pipeline cannot
fit each line on its own.

::::{container} d-figure
```{raw} html
:file: ../figures/start-resolution.svg
:class: wide-only
```

```{raw} html
:file: ../figures/start-resolution-narrow.svg
:class: narrow-only
```

:::{container} caption
A low-resolution spectrum blends the absorption lines that a high-resolution
spectrum resolves; the spectra are illustrative.
:::
::::

The blended spectrum still holds the information. DOROTHY uses a neural
network to read the labels from the whole spectrum. This approach follows
The Cannon {cite:p}`ness2015cannon`, The Payne {cite:p}`ting2019payne` and
astroNN {cite:p}`leung2019astronn`.

## Stars in common

Thousands of stars have both a low-resolution spectrum and high-resolution
labels. On these stars, the network learns to predict the labels from the
spectrum. It then predicts labels for the stars that have only a
low-resolution spectrum. This is label transfer.

::::{container} d-figure
```{raw} html
:file: ../figures/start-overlap.svg
:class: wide-only
```

```{raw} html
:file: ../figures/start-overlap-narrow.svg
:class: narrow-only
```

:::{container} caption
The network trains on the stars with both a spectrum and labels, and predicts
labels for the rest; the proportions are illustrative.
:::
::::

DOROTHY matches stars across surveys by their Gaia DR3 source ID
{cite:p}`gaia2023dr3`. It keeps every star in one file, the
{term}`super-catalogue`, with one row per star
([What is in the super-catalogue](../catalogue/contents.md)).

## One model for several surveys

Each survey samples the spectrum on its own wavelength grid, so one input
layer cannot read every survey. The multi-head model gives each survey its
own {term}`encoder`. An encoder maps the survey's spectrum to a vector of
fixed length. A shared {term}`trunk` combines the vectors of the surveys that
observed the star. One output {term}`head` per {term}`label source` then
predicts that source's labels.

::::{container} d-figure
```{raw} html
:file: ../figures/multi-head.svg
```

:::{container} caption
The multi-head model: one encoder per survey, a shared trunk, and one head per
label source.
:::
::::

A star needs a spectrum from only one of the surveys of the model. APOGEE and
GALAH measure the same quantities with different methods, so their labels
differ systematically. Separate heads keep the two scales apart
([The model](../methods/model.md)).

## Uncertainties

For every label, the network predicts two numbers: the prediction μ and the
{term}`predicted scatter` s.
Together they describe a normal distribution. The predicted scatter is the
expected spread of the measured labels around μ. It is large when the
spectrum constrains the label poorly, or when the label source's own
measurements are noisy.

::::{container} d-figure
```{raw} html
:file: ../figures/start-distribution.svg
:class: wide-only
```

```{raw} html
:file: ../figures/start-distribution-narrow.svg
:class: narrow-only
```

:::{container} caption
A prediction is a normal distribution with centre μ and width s; the values
are illustrative.
:::
::::

After training, DOROTHY checks the predicted scatter on stars that the
network did not train on. It fits one {term}`calibration factor` c per label
source and label, and reports the uncertainty c s
([Uncertainties](../methods/uncertainties.md)).
