# Use DOROTHY from Python

This guide runs the steps of the other guides from Python: configure, load
the data, train, evaluate and predict. The commands call the same functions.

## What you need

- DOROTHY, installed as in [Install](../start/install.md).
- A configuration file and the super-catalogue ([Train a model](train.md)).

## Step 1: Load the configuration

```python
import yaml

from dorothy.config import ExperimentConfig
from dorothy.data.training_data import complete_config_from_catalogue

with open("examples/variant2_multi_survey.yaml") as f:
    config = ExperimentConfig(**complete_config_from_catalogue(yaml.safe_load(f)))
```

{py:func}`~dorothy.data.training_data.complete_config_from_catalogue` fills in
the number of pixels of each survey, the output heads and the parameter names
from the catalogue, as `dorothy train` does.
{py:class}`~dorothy.config.schema.ExperimentConfig` validates the result.

## Step 2: Load the data and split it

```python
from dorothy.data.training_data import load_training_data

data = load_training_data(config)
data.configure_model(config)
split = data.split(config.data.train_ratio, config.data.val_ratio, config.seed)
```

{py:func}`~dorothy.data.training_data.load_training_data` loads the stars that
the configuration selects. `configure_model` sets the input size of a
single-survey MLP. `split` returns the rows of the training, validation and
test sets as a {py:class}`~dorothy.data.training_data.DataSplit`.

## Step 3: Train and save the checkpoint

```python
from dorothy.training import Trainer

trainer = Trainer(config)
history = data.fit(trainer, split)
checkpoint_dir = trainer.save_checkpoint()
split.save(checkpoint_dir, data)
```

`history` holds the losses and metrics of each epoch.
`save_checkpoint` writes the checkpoint directory, as `dorothy train` does,
and `split.save` adds the split, which evaluation reads.

## Step 4: Evaluate

```python
from dorothy.inference import evaluate_model, load_checkpoint

ckpt = load_checkpoint(checkpoint_dir)
results = evaluate_model(
    ckpt.model, ckpt.normalizer, ckpt.config, data, split.test,
    calibration=ckpt.calibration,
)
print(results["apogee"].physical.summary())
```

{py:func}`~dorothy.inference.checkpoint.load_checkpoint` rebuilds the model,
the label normalizer and the configuration.
{py:func}`~dorothy.inference.evaluation_utils.evaluate_model` returns one
{py:class}`~dorothy.inference.evaluation_utils.HeadEvaluation` per label
source, with the metrics in normalized and physical units. It applies the
calibration factors only when it is given them.

## Step 5: Prepare your own spectra

```python
import numpy as np

from dorothy.data import encode_precision, ivar_to_error_and_mask, normalize_spectra

flux, ivar = normalize_spectra(flux, ivar)  # each of shape (n, n_wavelengths)
sigma, mask = ivar_to_error_and_mask(ivar)
X = encode_precision(np.stack([flux, sigma, mask], axis=1))
```

The model takes spectra of shape `(n, 3, n_wavelengths)`, preprocessed as in
training ([Preprocessing](../methods/preprocessing.md)). This code does what
`dorothy predict` does with a FITS file. If the checkpoint's
`data.normalize_spectra` is false, call
{py:func}`~dorothy.data.fits_loader.zero_invalid_pixels` in place of
`normalize_spectra`. For a catalogue star,
{py:meth}`~dorothy.data.catalogue_loader.CatalogueLoader.load_spectrum` does
all of this.

## Step 6: Predict

```python
from dorothy.inference import Predictor

predictor = Predictor.load(checkpoint_dir, survey="boss")
result = predictor.predict(X, denormalize=True)
result.predictions, result.uncertainties  # each of shape (n, n_parameters)
```

{py:class}`~dorothy.inference.predictor.Predictor` takes spectra of one
survey, set with `survey=` for a multi-survey model, and `label_source=`
picks the head of a multi-head model. With `denormalize=True` the outputs are
in physical units; the default keeps {term}`normalized units`. The
uncertainties include the calibration factors when the checkpoint has them;
`Predictor.load(..., calibrate=False)` gives the raw predicted scatter, and
`result.calibrated` says which. `predict` applies no preprocessing.

To predict from several surveys at once, pass a dict of arrays, one per
survey, with `has_data`, a dict of `(n,)` booleans that marks which stars
have each survey.

## Next

- [API reference](../reference/api.md): every function and class.
- [Code map](../contribute/code-map.md): where each step lives in the code.
