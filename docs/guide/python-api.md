# Python API

The command-line tools are thin wrappers around these functions. The same
steps in Python:

```python
import yaml

from dorothy.config import ExperimentConfig
from dorothy.data.training_data import (
    complete_config_from_catalogue,
    load_training_data,
)
from dorothy.inference import Predictor, evaluate_model, load_checkpoint
from dorothy.training import Trainer

# Configuration, completed from the catalogue as `dorothy train` does
with open("examples/variant2_multi_survey.yaml") as f:
    config = ExperimentConfig(**complete_config_from_catalogue(yaml.safe_load(f)))

# Data, model size and split
data = load_training_data(config)
data.configure_model(config)
split = data.split(config.data.train_ratio, config.data.val_ratio, config.seed)

# Training and checkpoint
trainer = Trainer(config)
data.fit(trainer, split)
checkpoint_dir = trainer.save_checkpoint()
split.save(checkpoint_dir, data)

# Metrics on the test split, one entry per label source, with the
# calibrated uncertainties
ckpt = load_checkpoint(checkpoint_dir)
results = evaluate_model(
    ckpt.model, ckpt.normalizer, ckpt.config, data, split.test,
    calibration=ckpt.calibration,
)
print(results["apogee"].physical.summary())

# Predictions for new BOSS spectra X of shape (n, 3, n_wavelengths),
# preprocessed like the training spectra
predictor = Predictor.load(checkpoint_dir, survey="boss")
result = predictor.predict(X, denormalize=True)
result.predictions, result.uncertainties  # each (n, n_parameters), physical units
```

## Main objects

| Object | Role |
|--------|------|
| {py:class}`~dorothy.config.schema.ExperimentConfig` | The validated configuration |
| {py:func}`~dorothy.data.training_data.load_training_data` | Loads the stars a configuration selects, as {py:class}`~dorothy.data.training_data.TrainingData` |
| {py:class}`~dorothy.data.training_data.DataSplit` | Training, validation and test rows; `save` and `load` with a checkpoint |
| {py:class}`~dorothy.training.trainer.Trainer` | Trains a model; `history` holds the losses and metrics per epoch |
| {py:func}`~dorothy.inference.checkpoint.load_checkpoint` | Rebuilds the model, normalizer and configuration of a checkpoint |
| {py:func}`~dorothy.inference.evaluation_utils.evaluate_model` | Metrics and predictions per label source, as {py:class}`~dorothy.inference.evaluation_utils.HeadEvaluation` |
| {py:class}`~dorothy.inference.predictor.Predictor` | Predictions for arrays of spectra, and embeddings |
| {py:class}`~dorothy.data.catalogue_loader.CatalogueLoader` | Direct access to the catalogue, for example one star's spectrum |

## Prediction details

{py:meth}`~dorothy.inference.predictor.Predictor.predict` takes an array of
shape `(n, 3, n_wavelengths)` for a single survey (set with `survey=` when
loading a multi-survey model), or a dict of such arrays per survey together
with `has_data`, a dict of `(n,)` booleans. `label_source=` selects the head
of a multi-label model. With `denormalize=False` (the default) the outputs
stay in normalized units. A predictor from
{py:meth}`~dorothy.inference.predictor.Predictor.load` multiplies the
predicted scatter by the checkpoint's calibration factors for its head, when
the checkpoint has them; `calibrate=False` gives the raw scatter, and
`PredictionResult.calibrated` says which.

Spectra must be preprocessed as in training: 3 channels, invalid pixels set
to 0, the per-spectrum normalization when the checkpoint's
`data.normalize_spectra` is true, and then the precision encoding
({py:func}`~dorothy.data.fits_loader.encode_precision`) when
`data.precision_encoding` is true. `Predictor.predict` applies none of these
steps.
{py:meth}`~dorothy.data.catalogue_loader.CatalogueLoader.load_spectrum`, given
these two settings, does this for a catalogue star, and
{py:func}`~dorothy.inference.predictor.predict_from_fits` for a FITS file.
