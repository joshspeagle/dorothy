# Configuration

This page lists the rules for a configuration file, its sections and the
example files. Each section's page in the API reference lists every field with
its type, default and description.

## Rules

- A configuration is one YAML file, loaded into
  {py:class}`~dorothy.config.schema.ExperimentConfig`.
- An unknown key is an error, so a misspelled setting fails when the file
  loads.
- `data.survey` and `data.label_source` are aliases of `data.surveys` and
  `data.label_sources`. A string becomes a list of one.
- `dorothy train` fills in `multi_head_model.survey_wavelengths`,
  `multi_head_model.label_sources` and `data.parameter_names` from the
  catalogue. A value that you set for `data.parameter_names`,
  `multi_head_model.n_parameters` or `model.output_features` must match the
  catalogue's parameters.
- The checkpoint stores the completed configuration as `config.yaml`.
  `dorothy evaluate`, `dorothy predict` and `dorothy info` read it there.

## Model sections

A configuration has `model` or `multi_head_model`, not both.

| Data | Section | Without a model section |
|------|---------|-------------------------|
| One survey, one label source | `model` or `multi_head_model` | `model` with its defaults |
| Several surveys or several label sources | `multi_head_model` | `dorothy train` adds `multi_head_model` with its defaults |

`dorothy train` adds `multi_head_model` for several surveys or label
sources, so a `model` section there is an error. In Python,
`ExperimentConfig` with several label sources and no `multi_head_model` is an
error, and with several surveys and `model` it warns.

## Checks across fields

| Rule | Fields |
|------|--------|
| `train_ratio + val_ratio < 1`, so the test set is not empty | `data` |
| The keys name configured label sources | `training.label_source_weights`, `multi_head_model.head_dropout` |
| `ema_cpu_every` needs `ema_decay` | `training` |
| `p_*_min <= p_*_max`, and `f_min_override <= f_max` | `label_masking`, `input_masking` |
| `base_lr <= max_lr` for the `cyclic` schedule | `training.scheduler` |
| Training rejects `false`; the value exists to load [old checkpoints](old-checkpoints.md) | `data.precision_encoding`, `model.label_aware_uncertainty`, `multi_head_model.label_aware_uncertainty` |

## Sections

| Key | Fields | Sets |
|-----|--------|------|
| `name`, `description`, `output_dir`, `seed`, `device` | {py:class}`~dorothy.config.schema.ExperimentConfig` | The run's name, output directory, random seed and device |
| `data` | {py:class}`~dorothy.config.schema.DataConfig` | Catalogue, surveys, label sources, label quality cut, split, spectrum preprocessing |
| `model` | {py:class}`~dorothy.config.schema.ModelConfig` | The MLP for one survey and one label source |
| `multi_head_model` | {py:class}`~dorothy.config.schema.MultiHeadModelConfig` | Encoders, trunk and output heads ([The model](../methods/model.md)) |
| `training` | {py:class}`~dorothy.config.schema.TrainingConfig` | Epochs, batches, loss, warm-up, weight averaging, calibration, loss weights of label sources, mixed precision, saved epochs |
| `training.optimizer` | {py:class}`~dorothy.config.schema.OptimizerConfig` | Optimizer and weight decay |
| `training.scheduler` | {py:class}`~dorothy.config.schema.SchedulerConfig` | Learning-rate schedule |
| `label_masking` | {py:class}`~dorothy.config.schema.LabelMaskingConfig` | Random hiding of label sources and labels during training |
| `input_masking` | {py:class}`~dorothy.config.schema.InputMaskingConfig` | Random hiding of surveys and spectral blocks during training |

## Example files

The files in `examples/` use the catalogue at `data/super_catalogue.h5`,
except `variant6`.

| File | Surveys | Label sources | Model | Masking |
|------|---------|---------------|-------|---------|
| `boss_training.yaml` | BOSS | APOGEE | `model`, 10 epochs | no |
| `variant1_boss_apogee.yaml` | BOSS | APOGEE | `model` | no |
| `variant2_multi_survey.yaml` | BOSS, DESI | APOGEE | `multi_head_model` | no |
| `variant3_multi_labelset.yaml` | LAMOST LRS, LAMOST MRS | APOGEE, GALAH | `multi_head_model` | yes |
| `variant4_all_surveys.yaml` | all four | APOGEE | `multi_head_model` | no |
| `variant5_all_surveys_masked.yaml` | all four | APOGEE | `multi_head_model` | yes |
| `variant6_galah_xh_extended.yaml` | all four | APOGEE, GALAH | `multi_head_model`, `per_source` loss weighting | yes |

`variant6` reads `data/super_catalogue_xh_extended.h5`, the catalogue with 20
parameters and [X/H] abundances. Its header comment gives the builder command.
