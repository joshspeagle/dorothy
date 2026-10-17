# Quick start

This page trains a BOSS + DESI model on APOGEE labels and uses it.

## 1. Build the catalogue

Put the cross-matched FITS files in `data/raw_catalogues/`, then run:

```bash
python scripts/build_catalogue_v2.py
```

The builder writes `data/super_catalogue.h5`. The
[Super-catalogue](super_catalogue.md) page describes the inputs, the options
and the file layout.

## 2. Train

```bash
dorothy train examples/variant2_multi_survey.yaml
```

The configuration names the surveys (`boss`, `desi`), the label source
(`apogee`), the model and the training settings
([Configuration](configuration.md)). Training prints the losses per epoch and
writes a checkpoint directory:

```
outputs/variant2_multi_survey/variant2_multi_survey_final/
```

[Training](training.md) lists the files in it.

## 3. Evaluate

```bash
dorothy evaluate outputs/variant2_multi_survey/variant2_multi_survey_final
```

This prints the metrics of each output head on the test split, in normalized
and physical units, and writes plots to `<checkpoint>/plots/evaluation/`.

## 4. Predict

```bash
dorothy predict \
    --checkpoint outputs/variant2_multi_survey/variant2_multi_survey_final \
    --input boss_spectra.fits --survey boss --output predictions.csv
```

The CSV has one row per spectrum, with each parameter and its uncertainty
(`teff`, `teff_err`, ...) in physical units.

## The same steps in Python

```python
import yaml

from dorothy.config import ExperimentConfig
from dorothy.data.training_data import (
    complete_config_from_catalogue,
    load_training_data,
)
from dorothy.inference import Predictor, evaluate_model, load_checkpoint
from dorothy.training import Trainer

with open("examples/variant2_multi_survey.yaml") as f:
    config = ExperimentConfig(**complete_config_from_catalogue(yaml.safe_load(f)))

data = load_training_data(config)
data.configure_model(config)
split = data.split(config.data.train_ratio, config.data.val_ratio, config.seed)

trainer = Trainer(config)
data.fit(trainer, split)
checkpoint_dir = trainer.save_checkpoint()
split.save(checkpoint_dir, data)

# Metrics on the test split, per output head
ckpt = load_checkpoint(checkpoint_dir)
results = evaluate_model(ckpt.model, ckpt.normalizer, ckpt.config, data, split.test)
print(results["apogee"].physical.summary())

# Predictions for new BOSS spectra X, shape (n, 3, n_wavelengths) with
# channels [flux, sigma, mask], preprocessed like the training spectra
predictor = Predictor.load(checkpoint_dir, survey="boss")
result = predictor.predict(X, denormalize=True)
result.predictions, result.uncertainties  # each (n, n_parameters)
```

`complete_config_from_catalogue` fills in the survey wavelength counts, the
output heads and the parameter names from the catalogue, as `dorothy train`
does.
