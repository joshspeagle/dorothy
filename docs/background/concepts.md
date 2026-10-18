# Concepts

## Labels from spectra

A star's spectrum encodes its atmospheric parameters and composition: the
effective temperature (Teff), the surface gravity (log g), the iron abundance
[Fe/H] and the abundances of other elements relative to iron, [X/Fe], or to
hydrogen, [X/H]. Together these are the star's {term}`labels <label>`.

High-resolution surveys measure labels precisely. APOGEE observes the H band
(1.51–1.70 µm) at a resolving power of about 22,500 {cite:p}`majewski2017apogee`,
and its pipeline fits synthetic spectra to each observation
{cite:p}`garciaperez2016aspcap`. GALAH observes four optical bands at a
resolving power of about 28,000 {cite:p}`buder2021galahdr3`. Each catalogue
reports, for every label, a value, an uncertainty and quality flags.

Lower-resolution surveys observe many more stars. DESI {cite:p}`desi2022instrument`,
BOSS {cite:p}`dawson2013boss` and LAMOST {cite:p}`cui2012lamost` record
millions of stellar spectra at resolving powers between about 1500 and 7500,
too low to fit individual lines as the high-resolution pipelines do.

DOROTHY transfers the labels from the high-resolution surveys to the
lower-resolution ones. Thousands of stars have been observed by both kinds of
survey. A neural network learns, on these stars, to predict the
high-resolution labels from the lower-resolution spectrum, and then predicts
labels for stars that only the lower-resolution survey observed. This is a
data-driven approach in the line of The Cannon {cite:p}`ness2015cannon`,
The Payne {cite:p}`ting2019payne` and astroNN {cite:p}`leung2019astronn`.

## Stars in common

Stars are matched across surveys by their Gaia DR3 source ID
{cite:p}`gaia2023dr3`. The cross-matched files for each pair of a survey and a
label source go into the {term}`super-catalogue`, with one row per star. A
star can have spectra from several surveys and labels from both label sources,
or any subset of them ([The super-catalogue](../methods/catalogue.md)).

## One model for several surveys

The surveys sample the spectrum on different wavelength grids, so a network
cannot read their spectra with the same input layer. A multi-head model gives
each survey its own {term}`encoder`, which maps the survey's spectrum to a
vector of fixed length. A shared {term}`trunk` combines the vectors of
whatever surveys a star has, and one output {term}`head` per label source
predicts that source's labels ([Models](../methods/models.md)).

Two choices follow from this design:

- A star needs a spectrum from only one of the configured surveys. Training
  can randomly hide surveys a star does have, so the model learns to predict
  from any subset ([Augmentation](../methods/training.md#augmentation)).
- APOGEE and GALAH measure the same quantities with different methods, so
  their labels differ systematically. Separate heads keep the two scales apart
  while the trunk learns one description of the star.

## Uncertainties

The network predicts, for every label, a value μ and a
{term}`predicted scatter` s. Training maximizes the likelihood of the measured
label y under a normal distribution with mean μ and variance s²
([Loss](../methods/training.md#the-loss)). μ depends on the spectrum alone.
s can also read the label's reported uncertainty σ_y; prediction replaces σ_y
with a learned {term}`missing vector`. The predicted scatter is the expected
spread of the measured labels around μ, label noise included: it is large
where the spectrum constrains the label poorly or the labels are noisy. This
is heteroscedastic regression {cite:p}`nix1994mean,kendall2017uncertainties`.

The predictions are calibrated when the {term}`z-scores <z-score>`
z = (μ − y) / s of held-out stars have unit scatter
([Calibration](../methods/evaluation.md#calibration)). After training, one
{term}`calibration factor` per head and parameter rescales s so that the
z-scores of the validation stars have unit scatter, and prediction reports
the rescaled uncertainty.

## Normalized units

Spectra differ in flux scale between observations, and labels span very
different ranges (thousands of kelvin for Teff, tenths of a dex for
abundances). DOROTHY therefore scales each spectrum by its own median and
interquartile range, and each label by its median and interquartile range over
the training stars, with Teff in log10 ([Preprocessing](../methods/preprocessing.md)).
The model predicts in these {term}`normalized units`; predictions are converted
back to kelvin and dex at the end.
