# Configuration

This page lists the rules for configuration files and what each section
sets. The {py:mod}`configuration reference <dorothy.config.schema>` lists
every field with its type, default and allowed range.

## Rules

- Unknown keys are errors, so a misspelled setting fails when the file is
  loaded instead of being ignored.
- `survey` and `label_source` are accepted as aliases of `surveys` and
  `label_sources`.
- `dorothy train` fills in the survey wavelength counts
  (`multi_head_model.survey_wavelengths`), the output heads
  (`multi_head_model.label_sources`) and `data.parameter_names` from the
  catalogue. Parameter names that you set must match the catalogue.
- The checkpoint stores the completed configuration as `config.yaml`.
  `evaluate`, `predict` and `info` read it from there.

## Sections

`data`
:   The catalogue, the surveys and label sources, label quality cuts and the
    train/validation split. `max_flag_bits: 0` keeps only stars without any
    flagged parameter; `-1` keeps every unflagged label and masks the
    flagged ones individually. `normalize_spectra` applies the per-spectrum
    flux normalization (see [Preprocessing](../methods/preprocessing.md#spectra)).

`model` or `multi_head_model`
:   `model` is a plain MLP for one survey and one label source.
    `multi_head_model` is required for several surveys or label sources: one
    encoder per survey, a shared trunk and one output head per label source
    (see [The model](../methods/model.md)). A single-survey configuration can
    also use `multi_head_model`.

`training`
:   Epochs, batch size, loss, optimizer, learning-rate schedule, gradient
    clipping, early stopping, mixed precision (`amp`, `amp_dtype`), weight
    averaging (`ema_decay`, `ema_cpu_every`, see
    [Training](../methods/training.md#weight-averaging)), the warm-up of the
    loss (`nll_warmup_fraction`, see
    [Uncertainties](../methods/uncertainties.md#warm-up)), the fraction of
    stars trained without their label errors (`label_error_dropout`), the
    calibration factors fit after training (`calibrate_uncertainties`) and,
    with several label sources, how their losses combine
    (`label_source_weighting`, `label_source_weights`).

`label_masking`, `input_masking`
:   Training-time augmentation that randomly hides labels, label sources,
    spectral blocks and whole surveys (see
    [Training](../methods/training.md#augmentation)).
