# What is in the super-catalogue

The super-catalogue is one HDF5 file with every star's spectra and labels.
Each row is one star, identified by its Gaia DR3 source ID. For each survey,
the star's row holds its spectrum, or zeros if the survey did not observe it;
for each label source, its labels, or zeros if the source did not measure
them.

```{raw} html
:file: ../figures/catalogue.svg
```

## Layout

```
super_catalogue.h5
├── metadata/
│   ├── gaia_id          (N,) int64, sorted; the row's star
│   ├── ra, dec          (N,) float64, degrees (0 where unknown)
├── surveys/
│   ├── desi/, boss/, lamost_lrs/
│   │   ├── flux         (N, n_wavelengths) float32, 0 where missing
│   │   ├── ivar         (N, n_wavelengths) float32, 0 where missing or masked
│   │   ├── wavelength   (n_wavelengths,) float32, Å
│   │   └── snr          (N,) float32, 0 where missing
│   └── lamost_mrs/
│       ├── spectra      (N, 4, max(n_b, n_r)) float32: flux_b, ivar_b, flux_r, ivar_r
│       ├── wavelength_b (n_b,) float32, Å
│       ├── wavelength_r (n_r,) float32, Å
│       └── snr          (N,) float32
└── labels/
    ├── apogee/, galah/
    │   ├── values       (N, P) float32, 0 where missing
    │   ├── errors       (N, P) float32, 0 where missing or not measured
    │   ├── flags        (N, P) uint8, 0 = no warning
    │   └── gaia_id      (N,) int64, same as metadata/gaia_id
```

The file attributes are `n_stars`, `parameter_names`, `abundance_scale`
(`x_fe` or `x_h`), `survey_names`, `label_sources`, `creation_date` and
`version`. The LAMOST MRS arms are padded to the longer arm; the loaders trim
each arm to its wavelength grid and concatenate them, blue then red.
