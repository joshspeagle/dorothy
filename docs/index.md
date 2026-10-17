# DOROTHY

DOROTHY infers stellar parameters, chemical abundances and their uncertainties
from spectra with neural networks.

- **Inputs:** spectra from DESI, BOSS, LAMOST LRS and LAMOST MRS, alone or
  combined. A multi-survey model has one encoder per survey.
- **Labels:** APOGEE and GALAH parameters. A model has one output head per
  label source.
- **Uncertainties:** each prediction comes with a predicted scatter. A
  heteroscedastic loss trains it together with the measured label errors.

All data come from one HDF5 file, the super-catalogue. Each row is one star,
identified by its Gaia DR3 source ID, with the spectra of every survey that
observed it and the labels of every label source that measured it.

## Workflow

1. `scripts/build_catalogue_v2.py` builds the super-catalogue from
   cross-matched FITS files ([Super-catalogue](super_catalogue.md)).
2. A YAML file selects the surveys, label sources, model and training
   settings ([Configuration](configuration.md)).
3. `dorothy train` trains a model and writes a checkpoint directory
   ([Training](training.md)).
4. `dorothy evaluate` scores the checkpoint on its test split, and
   `dorothy predict` predicts for new spectra
   ([Evaluation and prediction](evaluation.md)).

The [Quick start](quickstart.md) runs these steps once. The
[API reference](api/index.md) documents every public module.
