# Read the catalogue in Python

{py:class}`~dorothy.data.catalogue_loader.CatalogueLoader` reads the file.
Training code does not call it directly:
{py:func}`~dorothy.data.training_data.load_training_data` picks the loader a
configuration needs ([Training](../methods/training.md#data-selection-and-split)).

```python
from dorothy.data import CatalogueLoader

loader = CatalogueLoader("data/super_catalogue.h5")
loader.info()                          # summary of surveys and label sources
names = loader.get_parameter_names()   # label column order

# The preprocessing of the default training configuration
inputs = {"normalize_spectra": True, "precision_encoding": True}

# One survey and one label source, stars with both, as model inputs
X, y = loader.load_for_training(survey="desi", label_source="apogee",
                                max_flag_bits=0, **inputs)

# Several surveys and label sources, spectra stored only where they exist
data = loader.load_merged_sparse(["boss", "desi", "lamost_mrs"],
                                 label_sources=["apogee", "galah"],
                                 max_flag_bits=-1, **inputs)

# One star's spectrum as a model input
X_star, wavelength, gaia_id = loader.load_spectrum("lamost_mrs", 123, **inputs)

# One label source for every star, [values, errors, mask], no spectra
y_galah = loader.load_labels("galah", max_flag_bits=0)
```

With several surveys, most stars lack most surveys. The sparse loader
({py:class}`~dorothy.data.catalogue_loader.SparseMergedData`) reads and keeps
each survey's spectra only for the stars that have them, with an index from
catalogue rows to stored rows. With several label sources, the dense
alternative (`data.use_dense_loading: true`) stores full arrays, zeros
included, and needs several times more memory. With one label source the
setting has no effect.
