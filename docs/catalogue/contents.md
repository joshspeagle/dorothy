# What is in the super-catalogue

The {term}`super-catalogue` is one HDF5 file. Each row is one star, matched
across surveys by its Gaia DR3 source ID {cite:p}`gaia2023dr3`. A row holds
the star's spectra and its labels, wherever they exist.

## One row per star

::::{container} d-figure
```{raw} html
:file: ../figures/catalogue.svg
:class: wide-only
```

```{raw} html
:file: ../figures/catalogue-narrow.svg
:class: narrow-only
```

:::{container} caption
A row is a star, its spectra and labels sit in column groups, and most cells
are empty; the stars are illustrative.
:::
::::

The rows are sorted by Gaia DR3 source ID, and row *i* is the same star in
every group of the file. Most stars appear in only one or two surveys.

Where a survey did not observe a star, the star's spectrum for that survey is
empty: flux 0, inverse variance 0 and S/N 0. All its pixels count as missing.
Where a label source did not measure a star, the star's labels from that
source are empty: value 0, error 0 and flag 0. All these labels count as
missing.

## The groups

| Group | Contents | One entry per star |
|-------|----------|--------------------|
| `metadata/` | Gaia DR3 source ID, RA and Dec | three numbers |
| `surveys/desi/`, `surveys/boss/`, `surveys/lamost_lrs/` | flux, inverse variance and S/N of one spectrum, and the survey's wavelength grid | one spectrum |
| `surveys/lamost_mrs/` | flux and inverse variance of the blue and red arms, S/N, and one wavelength grid per arm | two arms |
| `labels/apogee/`, `labels/galah/` | value, error and quality flag of every parameter | one label per parameter |

## The datasets

N is the number of stars, P the number of parameters, and `n_wavelengths`
the number of pixels of a survey's spectrum.

```text
super_catalogue.h5
├── metadata/
│   ├── gaia_id       (N,)        int64, sorted
│   └── ra, dec       (N,)        float64, degrees
├── surveys/
│   ├── desi/, boss/, lamost_lrs/
│   │   ├── flux        (N, n_wavelengths)  float32
│   │   ├── ivar        (N, n_wavelengths)  float32
│   │   ├── wavelength  (n_wavelengths,)    float32, Å
│   │   └── snr         (N,)                float32
│   └── lamost_mrs/
│       ├── spectra       (N, 4, n_max)  float32
│       ├── wavelength_b  (n_b,)         float32, Å
│       ├── wavelength_r  (n_r,)         float32, Å
│       └── snr           (N,)           float32
└── labels/
    └── apogee/, galah/
        ├── values   (N, P)  float32
        ├── errors   (N, P)  float32
        ├── flags    (N, P)  uint8
        └── gaia_id  (N,)    int64
```

An inverse variance of 0 marks a missing pixel. RA and Dec are 0 where they
are unknown. A label group's `gaia_id` equals `metadata/gaia_id`.
[Labels and quality flags](labels.md) explains the parameters, their units and
the flags. [How the catalogue is built](build.md#repeat-observations) defines
the S/N.

## The LAMOST MRS arms

A LAMOST MRS spectrum has a blue arm with `n_b` pixels and a red arm with
`n_r` pixels. The file stores both arms in one `spectra` array, with the
channels [flux_b, ivar_b, flux_r, ivar_r]. Each arm is padded with zeros to
the longer arm, `n_max` = max(`n_b`, `n_r`) pixels.
{py:class}`~dorothy.data.catalogue_loader.CatalogueLoader` cuts each arm to its wavelength grid and joins them, blue then red, into one
spectrum of `n_b + n_r` pixels.

## File attributes

| Attribute | Value |
|-----------|-------|
| `n_stars` | N, the number of rows |
| `parameter_names` | the P parameter names, in column order |
| `abundance_scale` | `x_fe` for [X/Fe] abundances, `x_h` for [X/H] |
| `survey_names` | `boss`, `desi`, `lamost_lrs`, `lamost_mrs` |
| `label_sources` | `apogee`, `galah` |
| `creation_date` | the build time, in ISO 8601 format |
| `version` | the version of the file layout, `2.0.0` |

[Read the catalogue in Python](python.md) shows how to load each group.
