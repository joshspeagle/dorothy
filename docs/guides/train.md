# Train a model

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

The files in `examples/` cover these setups:

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
