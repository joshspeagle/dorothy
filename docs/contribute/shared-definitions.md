# Shared definitions

Each of these is implemented once and used everywhere; change it there, not
in a caller:

| Definition | Implementation |
|------------|----------------|
| Which stars are used, and the split | {py:mod}`dorothy.data.training_data` |
| Spectrum preprocessing | {py:func}`~dorothy.data.fits_loader.zero_invalid_pixels`, {py:func}`~dorothy.data.fits_loader.normalize_spectra`, {py:func}`~dorothy.data.fits_loader.encode_precision` |
| Label normalization | {py:meth}`~dorothy.data.normalizer.LabelNormalizer.transform_labels` |
| Predicted scatter | {py:func}`~dorothy.losses.heteroscedastic.predicted_scatter` |
| Calibration of the predicted scatter | {py:func}`~dorothy.inference.calibration.calibrate_scatter` |
| Model forward pass per label source | {py:func}`~dorothy.training.batches.forward_label_sources` |

`tests/test_end_to_end.py` checks that evaluation reproduces the training
validation metrics for every data layout, which catches a definition that
drifted between two callers.
