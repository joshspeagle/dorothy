# Read the catalogue in Python

{py:class}`~dorothy.data.catalogue_loader.CatalogueLoader` reads the
super-catalogue. This page shows one task per code block, with the shapes each
call returns. N is the number of stars and P the number of parameters.

Training code does not call `CatalogueLoader` directly.
{py:func}`~dorothy.data.training_data.load_training_data` selects and splits
the stars for training
([Training](../methods/training.md#the-data-split)).

## Open the file

```python
from dorothy.data import CatalogueLoader

loader = CatalogueLoader("data/super_catalogue.h5")
loader.info()
```

`info()` prints, for each {term}`survey`, the number of stars with a spectrum
and the wavelength grid. It then prints the number of stars with
{term}`labels <label>` from each {term}`label source`, and the parameter
names.

## List the parameters

```python
names = loader.get_parameter_names()
```

`names` is the file's `parameter_names` attribute, a list of P strings in
label column order, such as `["teff", "logg", "fe_h", ...]`.

## Find a star's row

```python
import h5py
import numpy as np

source_id = 3165435934875840512  # an example Gaia DR3 source ID
with h5py.File("data/super_catalogue.h5", "r") as f:
    gaia_ids = f["metadata/gaia_id"][:]
row = int(np.flatnonzero(gaia_ids == source_id)[0])
```

`row` is the star's row in every group of the file. The last line raises an
`IndexError` if the star is not in the catalogue.

## Read one star's spectrum

```python
X, wavelength, gaia_id = loader.load_spectrum("lamost_mrs", row)
```

`X` has shape `(3, n_wavelengths)`, with the channels [flux, ρ, mask]
([Preprocessing](../methods/preprocessing.md) defines ρ). `wavelength` has
shape `(n_wavelengths,)`, in Å, and `gaia_id` is the star's source ID. For
LAMOST MRS the pixels are the blue arm, then the red arm. If the survey did
not observe the star, `X` is all zeros, so every pixel is masked. With
`normalize_spectra=True`, the flux is normalized as in training.

## Read one survey's spectra with labels

```python
data = loader.load(survey="desi", label_source="apogee", require_complete=True)
```

`data` is a {py:class}`~dorothy.data.catalogue_loader.CatalogueData` with the
stars that have a DESI spectrum and at least one APOGEE label error above 0.
It holds the arrays as stored: `data.flux` and `data.ivar` of shape
`(n_stars, n_wavelengths)`, `data.wavelength`, and `data.labels`,
`data.label_errors` and `data.label_flags` of shape `(n_stars, P)`.
`data.gaia_ids` gives each star's source ID.

## Inspect one survey as model inputs

```python
X, y = loader.load_for_training(
    "desi", label_source="apogee", max_flag_bits=0, normalize_spectra=True
)
```

`X` has shape `(n_stars, 3, n_wavelengths)` = [flux, ρ, mask], and `y` has
shape `(n_stars, 3, P)` = [values, errors, mask]. The stars are those of the
previous task, without the stars that have more flagged labels than
`max_flag_bits`. `normalize_spectra=True` matches the default training
configuration. Use this call to inspect one survey's inputs, not to train: it
does not split the stars.

## Read one label source's labels

```python
y = loader.load_labels("galah", max_flag_bits=0)
```

`y` has shape `(N, 3, P)` = [values, errors, mask], one row per catalogue
star, with no spectra. The {term}`mask` is 1 where a label is valid, and 0 for stars
that GALAH did not measure. [Labels and quality flags](labels.md#when-a-label-is-valid)
defines a valid label and `max_flag_bits`.

## Read several surveys sparsely

```python
data = loader.load_merged_sparse(
    ["boss", "desi", "lamost_mrs"],
    label_sources=["apogee", "galah"],
    max_flag_bits=0,
    normalize_spectra=True,
)
```

Most stars lack most surveys, so `data`, a
{py:class}`~dorothy.data.catalogue_loader.SparseMergedData`, stores each
survey's spectra only for the stars that have one. For each survey `s`:

- `data.flux[s]` and `data.sigma[s]` have shape `(n_with_data, n_wavelengths)`.
  `data.sigma[s]` holds ρ.
- `data.global_to_local[s]` has shape `(N,)`. It gives each catalogue row's
  position in `data.flux[s]`, or −1 if the star has no spectrum from `s`.
- `data.local_to_global[s]` has shape `(n_with_data,)` and maps back.

`data.labels_dict[source]` holds the labels of each label source, of shape
`(N, 3, P)`. The call prints its progress for each survey.
