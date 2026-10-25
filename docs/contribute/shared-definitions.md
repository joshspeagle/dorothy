# Shared definitions

This page lists the steps that training, evaluation and prediction must do
the same way. Each has one implementation. To change one, edit that
implementation; never repeat it in a caller.

## The definitions

| Definition | Implementation | Callers |
|------------|----------------|---------|
| Which stars are used, and the split | {py:func}`~dorothy.data.training_data.load_training_data`, {py:meth}`~dorothy.data.training_data.TrainingData.split` | `dorothy train`, `dorothy evaluate`, the Python API |
| Spectrum preprocessing | {py:func}`~dorothy.data.fits_loader.zero_invalid_pixels`, {py:func}`~dorothy.data.fits_loader.normalize_spectra`, {py:func}`~dorothy.data.fits_loader.encode_precision` | Every catalogue loader, `predict_from_fits` |
| Label normalization | {py:meth}`~dorothy.data.normalizer.LabelNormalizer.transform_labels` | Training, evaluation |
| Predicted scatter | {py:func}`~dorothy.losses.heteroscedastic.predicted_scatter` | The loss, validation, evaluation, prediction, saliency |
| Calibration of the predicted scatter | {py:func}`~dorothy.inference.calibration.calibrate_scatter` | Evaluation, prediction |
| Model forward pass per label source | {py:func}`~dorothy.training.batches.forward_label_sources` | Training, validation, evaluation |

## Data formats

Every function passes data in these shapes
([Preprocessing](../methods/preprocessing.md) defines them).

| Array | Shape | Channels |
|-------|-------|----------|
| Spectra | `(N, 3, n_wavelengths)` | flux, $\rho = \ln(1 + 1/\sigma)$, mask |
| Labels | `(N, 3, n_parameters)` | values, errors, mask |
| Model output | `(N, 2, n_parameters)` | $\mu$, $\ln s$, in normalized units |

- A masked entry can hold any value, NaN and infinity included. Select with
  the mask (`torch.where`, boolean indexing) before arithmetic; never
  multiply by the mask.
- The parameter names, and their number, come from the catalogue's
  `parameter_names` attribute. Never assume 11 parameters.
- Random numbers come from the configured `seed`:
  `numpy.random.RandomState(seed)` for the split, and seeded generators for
  augmentation and initialization.

## The test that guards them

`tests/test_end_to_end.py` checks that evaluation reproduces the validation
metrics of training for every data layout. A definition that
drifts between two callers fails this test.
