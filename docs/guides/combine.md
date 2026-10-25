# Combine surveys and label sources

This guide trains one model on spectra from several surveys and on labels
from both APOGEE and GALAH.

## What you need

- A configuration that trains, as in [Train a model](train.md).
- The super-catalogue, with the surveys and label sources you choose.

## Step 1: Choose the data

List them in the `data` section:

```yaml
data:
  catalogue_path: data/super_catalogue.h5
  surveys: [boss, desi, lamost_lrs, lamost_mrs]
  label_sources: [apogee, galah]
  max_flag_bits: -1
```

Training keeps a star when it has a spectrum from at least one of the
surveys. It must also have at least one valid label. Most stars have only some
surveys and labels; the model treats the rest as missing.

`max_flag_bits: -1` keeps each star's valid labels and masks only its flagged
ones. Stars with a few flagged labels are therefore not dropped
([Labels and quality flags](../catalogue/labels.md)).

## Step 2: Size the multi-head model

Several surveys or label sources need the multi-head model: one
{term}`encoder` per survey, a shared {term}`trunk`, and one output
{term}`head` per label source ([The model](../methods/model.md)).
`dorothy train` adds the `multi_head_model` section itself. It fills in the
number of pixels of each survey from the catalogue, and the heads from
`data.label_sources`.

Write the section only to change a value, such as the layer sizes:

```yaml
multi_head_model:
  latent_dim: 128
  encoder_hidden: [512, 256]
```

Every field that you leave out keeps its default
([Configuration](../reference/configuration.md)).

## Step 3: Turn on masking

```yaml
input_masking:
  enabled: true
label_masking:
  enabled: true
```

Input masking hides whole surveys and blocks of pixels of each training star,
afresh for every batch. Label masking hides label sources and single labels
in the same way. Both teach the model to predict from the incomplete data
that most stars have ([Training](../methods/training.md)). The fields of each
section set how much they hide
([Configuration](../reference/configuration.md)).

## Step 4: Weight the label sources

```yaml
training:
  label_source_weighting: per_source
  label_source_weights: {apogee: 1.0, galah: 0.5}
```

With the default, `per_label`, every valid label in a batch counts equally,
so the label source with more labels weighs more. `per_source` gives
each label source the same total weight, whatever its number of labels.
With either setting, `label_source_weights` multiplies the weight of each
source; a source that it leaves out keeps weight 1. [Uncertainties](../methods/uncertainties.md)
gives the loss.

## Step 5: Train

```bash
dorothy train combined.yaml
```

The output also prints the best epoch of each label source.

## Step 6: Use each head

The checkpoint works as in the other guides, with two differences:

- `dorothy evaluate` reports the metrics and plots of each head
  ([Evaluate a model](evaluate.md)).
- `dorothy predict` needs `--survey` for the spectra and takes
  `--label-source` for the head ([Predict labels for your
  spectra](predict.md)).

In `examples/`, `variant3_multi_labelset.yaml` and
`variant6_galah_xh_extended.yaml` are complete configurations of this kind.

## Next

- [Evaluate a model](evaluate.md).
- [Compare APOGEE and GALAH](compare-labels.md): check how the two label
  sources agree before you train on both.
