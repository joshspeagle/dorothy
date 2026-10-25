# Survey properties

This page lists the wavelength range, the resolution and the line spread
function (LSF) of each survey whose spectra DOROTHY reads. Each section links to
its sources.

## Overview

| Survey | Wavelengths | Resolution R | LSF representation |
|--------|-------------|--------------|--------------------|
| DESI | 3600–9800 Å | 2000–5000 | Resolution matrix per spectrum |
| BOSS | 3600–10,400 Å | 1560–2650 | Gaussian $\sigma$ per pixel (`WAVEDISP`) |
| LAMOST LRS | 3700–9000 Å | 1800 at 5500 Å | None published |
| LAMOST MRS | 4950–5350 Å and 6300–6800 Å | 7500 at 5163 Å and 6593 Å | None published |

The catalogue stores each survey on the wavelength grid of its input files
([How the catalogue is built](../catalogue/build.md)). DOROTHY does not use
the LSF.

## Resolution and line width

$$
\begin{aligned}
R &= \frac{\lambda}{\Delta\lambda} , \\
\sigma_\lambda &= \frac{\Delta\lambda}{2\sqrt{2\ln 2}} \approx \frac{\Delta\lambda}{2.355} , \\
\sigma_v &= \frac{c}{2.355\,R} \approx \frac{127{,}300\ \mathrm{km\,s^{-1}}}{R} .
\end{aligned}
$$

$\Delta\lambda$ is the full width at half maximum (FWHM) of the LSF at
wavelength $\lambda$. $\sigma_\lambda$ is the standard deviation of a Gaussian
with that FWHM, and $\sigma_v$ the same width as a velocity, with
$c = 299{,}792\ \mathrm{km\,s^{-1}}$. For example, $R = 1800$ gives
$\sigma_v \approx 71\ \mathrm{km\,s^{-1}}$, and $R = 7500$ gives
$\sigma_v \approx 17\ \mathrm{km\,s^{-1}}$.

## DESI

DESI has three spectrograph channels, whose wavelength ranges overlap
{cite:p}`desi2022instrument`.

| Channel | Wavelengths |
|---------|-------------|
| Blue (B) | 3600–5930 Å |
| Red (R) | 5600–7720 Å |
| Infrared (Z) | 7470–9800 Å |

The resolution rises with wavelength
([DESI spectrograph](https://www.desi.lbl.gov/spectrograph/)).

| Wavelengths | R |
|-------------|---|
| 3600–5550 Å | 2000–3200 |
| 5550–6560 Å | 3200–4100 |
| 6560–9800 Å | 4100–5000 |

Each spectrum has its own resolution matrix, stored as its diagonals in the
`B_RESOLUTION`, `R_RESOLUTION` and `Z_RESOLUTION` extensions of the coadd
files, one per channel.

## BOSS

BOSS has two channels, split by a dichroic at about 6000 Å
([BOSS spectrograph](https://www.sdss4.org/instruments/boss_spectrograph/)).

| Channel | Wavelengths (at least) | R |
|---------|------------------------|---|
| Blue | 3600–6350 Å | 1560 at 3700 Å to 2270 at 6000 Å |
| Red | 5650–10,000 Å | 1850 at 6000 Å to 2650 at 9000 Å |

The `WAVEDISP` extension (HDU 4) of the `spCFrame` files holds the LSF at
every pixel. It is the $\sigma$ of a fitted Gaussian, in pixels of the
log-wavelength grid. A pixel spans 10⁻⁴ in log₁₀ λ. The wavelengths are in vacuum.

## LAMOST LRS

LAMOST low-resolution spectra cover 3700–9000 Å with R = 1800 at 5500 Å
([DR9 data description](https://www.lamost.org/dr9/v2.0/doc/lr-data-production-description)). The pixels are log-linear, with a
spacing of 10⁻⁴ in log₁₀ λ, and the wavelengths are in vacuum. The data
release publishes no LSF.

## LAMOST MRS

LAMOST medium-resolution spectra have two arms: blue, 4950–5350 Å, and red,
6300–6800 Å. Each arm has R = 7500 at its reference wavelength: 5163 Å for
blue and 6593 Å for red
([DR7 data description](https://dr7.lamost.org/v2.0/doc/mr-data-production-description)).
The wavelengths are in vacuum. The data release publishes no LSF. The catalogue stores both arms
([File formats](file-formats.md#super-catalogue)).

## Sources

The values on this page come from these sources. The survey papers are
{cite:t}`desi2022instrument` (DESI), {cite:t}`smee2013boss` (BOSS) and
{cite:t}`cui2012lamost,liu2020lamostmrs` (LAMOST).

| Survey | Values | LSF data |
|--------|--------|----------|
| DESI | {cite:t}`desi2022instrument`, Figure 14 (channel ranges); [DESI spectrograph](https://www.desi.lbl.gov/spectrograph/); [Guy et al. 2023, arXiv:2209.14482](https://arxiv.org/abs/2209.14482) | [coadd file data model](https://desidatamodel.readthedocs.io/en/latest/DESI_SPECTRO_REDUX/SPECPROD/healpix/SURVEY/PROGRAM/PIXGROUP/PIXNUM/coadd-SURVEY-PROGRAM-PIXNUM.html) |
| BOSS | [BOSS spectrograph](https://www.sdss4.org/instruments/boss_spectrograph/) | [spCFrame data model](https://data.sdss.org/datamodel/files/BOSS_SPECTRO_REDUX/RUN2D/PLATE4/spCFrame.html) |
| LAMOST LRS | [LAMOST DR9 low-resolution data description](https://www.lamost.org/dr9/v2.0/doc/lr-data-production-description) | None |
| LAMOST MRS | [LAMOST DR7 medium-resolution data description](https://dr7.lamost.org/v2.0/doc/mr-data-production-description) | None |
