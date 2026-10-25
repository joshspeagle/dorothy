# Train a model

This guide writes a configuration file, trains a model on the
{term}`super-catalogue`, and finds the {term}`checkpoint` that the other guides
use.

## What you need

- DOROTHY, installed as in [Install](../start/install.md).
- The super-catalogue, an HDF5 file ([How the catalogue is
  built](../catalogue/build.md)).

## Step 1: Write a configuration

A configuration is a YAML file. This one trains on BOSS spectra with APOGEE
labels:

```yaml
name: boss_apogee
output_dir: outputs

data:
  catalogue_path: data/super_catalogue.h5
  surveys: [boss]
  label_sources: [apogee]

training:
  epochs: 100
```

`name` names the run and its output directory. `data` names the catalogue,
the surveys whose spectra go in, and the label sources whose labels the model
learns. Every field that the file leaves out takes its default. With one
survey, one label source and no model section, the model is the default MLP.
[Configuration](../reference/configuration.md) lists every field, with its
default and allowed values. A misspelled field is an error, not ignored.

The files in `examples/` are complete configurations to start from:

| File | Surveys | Label sources | Masking |
|------|---------|---------------|---------|
| `boss_training.yaml` | BOSS | APOGEE | no |
| `variant1_boss_apogee.yaml` | BOSS | APOGEE | no |
| `variant2_multi_survey.yaml` | BOSS, DESI | APOGEE | no |
| `variant3_multi_labelset.yaml` | LAMOST LRS, LAMOST MRS | APOGEE, GALAH | yes |
| `variant4_all_surveys.yaml` | all four | APOGEE | no |
| `variant5_all_surveys_masked.yaml` | all four | APOGEE | yes |
| `variant6_galah_xh_extended.yaml` | all four | APOGEE, GALAH | yes |

`boss_training.yaml` is a short run of 10 epochs. `variant6` needs a catalogue
built with 20 parameters as [X/H] (`data/super_catalogue_xh_extended.h5`).
[Combine surveys and label sources](combine.md) explains the settings of the
examples with several surveys or label sources.

## Step 2: Train

```bash
dorothy train boss_apogee.yaml
```

The command loads the catalogue, splits the stars into training, validation
and test sets, and trains. It prints the size of each set, a progress bar
with the training and validation losses, and the best epoch. `--device`
picks the device, and `--resume` starts from the weights of an earlier
checkpoint ([Commands](../reference/commands.md)).

## Step 3: Find the checkpoint

The run writes the checkpoint directory
`<output_dir>/<name>/<name>_final/`, here
`outputs/boss_apogee/boss_apogee_final/`. The other commands take this
directory. Three of its files matter most:

| File | Contents |
|------|----------|
| `best_model.pth` | the weights of the epoch with the lowest validation loss |
| `calibration.json` | the calibration factors of `best_model.pth` |
| `config.yaml` | the configuration, completed from the catalogue |

[Training](../methods/training.md) lists the other files. To check what the
checkpoint holds, print its summary:

```bash
dorothy info outputs/boss_apogee/boss_apogee_final
```

## Step 4: Look at the training plots

The run writes its plots to the `plots/` directory of the checkpoint.
`loss_curves.png` shows the training and validation losses per epoch, with
the best epoch marked, and the learning rate. If the validation loss rises
while the training loss falls, the model overfits; `best_model.pth` holds the
weights from before that rise.

## Next

- [Evaluate a model](evaluate.md) on its test stars.
- [Predict labels for your spectra](predict.md).
