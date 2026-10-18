# Configuration

An experiment is one YAML file, loaded into
{py:class}`~dorothy.config.schema.ExperimentConfig`. The
{py:mod}`configuration reference <dorothy.config.schema>` lists every field
with its type, default and allowed range. This page explains how the sections
fit together.

```yaml
name: multi_survey_experiment
output_dir: outputs
seed: 42
device: auto                 # auto, cuda, mps or cpu

data:
  catalogue_path: data/super_catalogue.h5
  surveys: [boss, desi]
  label_sources: [apogee]
  max_flag_bits: -1
  normalize_spectra: true

multi_head_model:
  latent_dim: 256
  encoder_hidden: [1024, 512]
  trunk_hidden: [512, 256]
  output_hidden: [64]
  combination_mode: concat

training:
  epochs: 100
  batch_size: 512
  learning_rate: 0.001
  loss: heteroscedastic
  optimizer:
    type: adamw
    weight_decay: 0.01
  scheduler:
    type: one_cycle
    max_lr: 0.001

label_masking:
  enabled: false
input_masking:
  enabled: false
```

The files `examples/variant*.yaml` cover the supported setups:

| File | Surveys | Label sources | Masking |
|------|---------|---------------|---------|
| `boss_training.yaml` | BOSS | APOGEE | no |
| `variant1_boss_apogee.yaml` | BOSS | APOGEE | no |
| `variant2_multi_survey.yaml` | BOSS, DESI | APOGEE | no |
| `variant3_multi_labelset.yaml` | LAMOST LRS, LAMOST MRS | APOGEE, GALAH | yes |
| `variant4_all_surveys.yaml` | all four | APOGEE | no |
| `variant5_all_surveys_masked.yaml` | all four | APOGEE | yes |
| `variant6_galah_xh_extended.yaml` | all four | APOGEE, GALAH | yes |

`variant6` uses the 20-parameter [X/H] catalogue
(`data/super_catalogue_xh_extended.h5`) and `per_source` loss weighting.

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
    (see [Models](../methods/models.md)). A single-survey configuration can
    also use `multi_head_model`.

`training`
:   Epochs, batch size, loss, optimizer, learning-rate schedule, gradient
    clipping, early stopping, mixed precision (`amp`), weight averaging
    (`ema_decay`, see [Training](../methods/training.md#weight-averaging))
    and, with several label sources, how their losses combine
    (`label_source_weighting`, `label_source_weights`).

`label_masking`, `input_masking`
:   Training-time augmentation that randomly hides labels, label sources,
    spectral blocks and whole surveys (see
    [Training](../methods/training.md#augmentation)).
