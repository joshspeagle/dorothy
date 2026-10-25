# Code map

This page shows where each step of the pipeline lives in the code, so that
you can find the file to read or change.

::::{container} d-figure
```{raw} html
:file: ../figures/pipeline.svg
```

:::{container} caption
The pipeline: the builder writes the catalogue, training writes a
checkpoint, and evaluation and prediction read it.
:::
::::

## Packages

| Package | What it does | Start reading at |
|---------|--------------|------------------|
| {py:mod}`dorothy.config` | Validates a configuration file | {py:class}`~dorothy.config.schema.ExperimentConfig` |
| {py:mod}`dorothy.data` | Reads the catalogue, selects and splits stars, preprocesses spectra and labels, masks inputs and labels during training | {py:func}`~dorothy.data.training_data.load_training_data`, {py:class}`~dorothy.data.catalogue_loader.CatalogueLoader` |
| {py:mod}`dorothy.models` | The networks and their output layers | {py:class}`~dorothy.models.mlp.MLP`, {py:class}`~dorothy.models.multi_head_mlp.MultiHeadMLP`, {py:class}`~dorothy.models.output.LabelAwareOutput` |
| {py:mod}`dorothy.losses` | The loss and the predicted scatter | {py:class}`~dorothy.losses.heteroscedastic.HeteroscedasticLoss`, {py:func}`~dorothy.losses.heteroscedastic.predicted_scatter` |
| {py:mod}`dorothy.training` | The training loop, batches and checkpoints | {py:class}`~dorothy.training.trainer.Trainer`, {py:func}`~dorothy.training.batches.forward_label_sources` |
| {py:mod}`dorothy.inference` | Loads checkpoints, predicts, evaluates, calibrates | {py:func}`~dorothy.inference.checkpoint.load_checkpoint`, {py:class}`~dorothy.inference.predictor.Predictor`, {py:func}`~dorothy.inference.evaluation_utils.evaluate_model` |
| {py:mod}`dorothy.analysis` | Saliency maps, anomaly detection, comparison of label sources | {py:class}`~dorothy.analysis.saliency.SaliencyAnalyzer`, {py:class}`~dorothy.analysis.knn_anomaly.AnomalyDetector`, {py:func}`~dorothy.analysis.label_comparison.compare_label_sources` |
| {py:mod}`dorothy.visualization` | Training and evaluation plots | {py:func}`~dorothy.visualization.evaluation_plots.generate_evaluation_plots` |
| {py:mod}`dorothy.cli` | The `dorothy` command | {py:func}`~dorothy.cli.main.main` |

Outside the package:

| Path | What it holds |
|------|---------------|
| `scripts/build_catalogue_v2.py` | The catalogue builder |
| `scripts/legacy/` | The deprecated v1 pipeline |
| `examples/` | Training configurations and saliency scripts |
| `tests/` | The test suite; `tests/synthetic.py` writes small catalogues with all four surveys and both label sources |
| `docs/` | These docs ([Build the docs](build-docs.md)) |

## Training

`dorothy train` runs {py:func}`~dorothy.cli.main.cmd_train`, which calls these
functions in order.

1. {py:func}`~dorothy.data.training_data.complete_config_from_catalogue` fills
   in the settings that come from the catalogue, and
   {py:class}`~dorothy.config.schema.ExperimentConfig` validates the result.
2. {py:func}`~dorothy.data.training_data.load_training_data` selects the
   stars and loads them in one of three layouts (below).
3. {py:meth}`~dorothy.data.training_data.TrainingData.split` divides the rows
   into training, validation and test sets with the configured seed.
4. {py:meth}`~dorothy.data.training_data.TrainingData.fit` passes the data to
   the `Trainer.fit*` method of its layout. All of these methods share one
   training loop.
5. {py:meth}`~dorothy.training.trainer.Trainer.save_checkpoint` and
   {py:meth}`~dorothy.data.training_data.DataSplit.save` write the checkpoint
   directory ([File formats](../reference/file-formats.md#checkpoint-directory)).

| Layout | Used for | Storage |
|--------|----------|---------|
| `single_survey` | One survey and one label source | Dense arrays |
| `dense_multi_label` | Several label sources with `data.use_dense_loading: true` | Dense arrays per survey and label source |
| `sparse_multi_survey` | Every other configuration | {py:class}`~dorothy.data.catalogue_loader.SparseMergedData` |

The training loop takes batches from
{py:class}`~dorothy.training.batches.TensorBatchSource` (dense arrays) or
{py:class}`~dorothy.training.batches.SparseBatchSource` (sparse storage). A
{py:class}`~dorothy.training.batches.Batch` holds a dict of spectra per
survey, a dict of `has_data` masks and a dict of labels per label source.

## Evaluation and prediction

`dorothy evaluate` loads the checkpoint with
{py:func}`~dorothy.inference.checkpoint.load_checkpoint`, reloads the data
with the same `load_training_data`, and scores the test rows with
{py:func}`~dorothy.inference.evaluation_utils.evaluate_model`. That function
runs the model through `forward_label_sources`, as training validation does.

`dorothy predict` runs
{py:func}`~dorothy.inference.predictor.predict_from_fits`: it reads the FITS
file with {py:class}`~dorothy.data.fits_loader.FITSLoader`, preprocesses the
spectra and predicts with {py:class}`~dorothy.inference.predictor.Predictor`.

## Where to make a change

| To change | Edit |
|-----------|------|
| A configuration setting | Its `Field` in `dorothy/config/schema.py`; the description there is its documentation |
| Which stars are used, or the split | `dorothy/data/training_data.py` |
| Spectrum or label preprocessing | `dorothy/data/fits_loader.py`, `dorothy/data/normalizer.py` |
| The network | `dorothy/models/` |
| The loss | `dorothy/losses/heteroscedastic.py` |
| The training loop | `dorothy/training/trainer.py` |
| The files of a checkpoint | `Trainer.save_checkpoint` and `dorothy/inference/checkpoint.py`; then [Old checkpoints](../reference/old-checkpoints.md) |
| A command or its options | `dorothy/cli/main.py`; then [Commands](../reference/commands.md) |
| The catalogue layout | `scripts/build_catalogue_v2.py`, `dorothy/data/catalogue_loader.py` and `tests/synthetic.py` |

Several of these steps have one shared implementation; [Shared
definitions](shared-definitions.md) lists them. To add a survey, follow
[Add a survey](add-a-survey.md).
