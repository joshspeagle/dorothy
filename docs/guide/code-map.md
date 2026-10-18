# Code map

This page shows where each step of the pipeline lives, so you can find the
code to read or change.

```{raw} html
:file: ../figures/pipeline.svg
```

## Packages

| Package | What it does | Main entry points |
|---------|--------------|-------------------|
| {py:mod}`dorothy.config` | Validates experiment configurations | {py:class}`~dorothy.config.schema.ExperimentConfig` |
| {py:mod}`dorothy.data` | Reads the catalogue, selects and splits stars, preprocesses spectra and labels, masks inputs during training | {py:func}`~dorothy.data.training_data.load_training_data`, {py:class}`~dorothy.data.catalogue_loader.CatalogueLoader`, {py:func}`~dorothy.data.fits_loader.normalize_spectra`, {py:class}`~dorothy.data.normalizer.LabelNormalizer` |
| {py:mod}`dorothy.models` | The networks | {py:class}`~dorothy.models.mlp.MLP`, {py:class}`~dorothy.models.multi_head_mlp.MultiHeadMLP` |
| {py:mod}`dorothy.losses` | Heteroscedastic and masked MSE losses | {py:class}`~dorothy.losses.heteroscedastic.HeteroscedasticLoss`, {py:func}`~dorothy.losses.heteroscedastic.predicted_scatter` |
| {py:mod}`dorothy.training` | The training loop and batch assembly | {py:class}`~dorothy.training.trainer.Trainer` |
| {py:mod}`dorothy.inference` | Checkpoints, prediction, evaluation | {py:func}`~dorothy.inference.checkpoint.load_checkpoint`, {py:class}`~dorothy.inference.predictor.Predictor`, {py:func}`~dorothy.inference.evaluation_utils.evaluate_model` |
| {py:mod}`dorothy.analysis` | Saliency maps, anomaly detection, comparison of label sources | {py:class}`~dorothy.analysis.saliency.SaliencyAnalyzer`, {py:class}`~dorothy.analysis.knn_anomaly.AnomalyDetector`, {py:func}`~dorothy.analysis.label_comparison.compare_label_sources` |
| {py:mod}`dorothy.visualization` | Training and evaluation plots | {py:func}`~dorothy.visualization.evaluation_plots.generate_evaluation_plots` |
| {py:mod}`dorothy.cli` | The `dorothy` command | {py:func}`~dorothy.cli.main.main` |

Outside the package:

- `scripts/build_catalogue_v2.py` builds the super-catalogue;
  `scripts/legacy/` holds the deprecated v1 pipeline.
- `examples/` holds training configurations and saliency scripts.
- `tests/synthetic.py` writes small catalogues with all four surveys and both
  label sources, which the end-to-end tests train on.

## How the pieces connect

`dorothy train` (in {py:func}`~dorothy.cli.main.cmd_train`) calls, in order:

1. {py:func}`~dorothy.data.training_data.complete_config_from_catalogue` and
   {py:class}`~dorothy.config.schema.ExperimentConfig`, for the configuration;
2. {py:func}`~dorothy.data.training_data.load_training_data`, which picks one
   of three layouts: a single survey in dense arrays, several surveys in dense
   arrays, or several surveys in sparse storage
   ({py:class}`~dorothy.data.catalogue_loader.SparseMergedData`);
3. {py:meth}`~dorothy.data.training_data.TrainingData.split` and
   {py:meth}`~dorothy.data.training_data.TrainingData.fit`, which hands the
   data to the matching `Trainer.fit*` method.

All `Trainer.fit*` methods share one loop. Batches come from
{py:class}`~dorothy.training.batches.TensorBatchSource` (dense arrays) or
{py:class}`~dorothy.training.batches.SparseBatchSource` (sparse storage), as
{py:class}`~dorothy.training.batches.Batch` objects: a dict of survey inputs,
a dict of `has_data` masks and a dict of labels per label source.
{py:func}`~dorothy.training.batches.forward_label_sources` runs the model on
a batch for training and evaluation alike.

`dorothy evaluate` loads the checkpoint with
{py:func}`~dorothy.inference.checkpoint.load_checkpoint`, reloads the data
with the same `load_training_data`, and scores the test rows with
{py:func}`~dorothy.inference.evaluation_utils.evaluate_model`.

## Shared definitions

Each of these is implemented once and used everywhere; change it there, not
in a caller:

| Definition | Implementation |
|------------|----------------|
| Which stars are used, and the split | {py:mod}`dorothy.data.training_data` |
| Spectrum preprocessing | {py:func}`~dorothy.data.fits_loader.normalize_spectra` |
| Label normalization | {py:meth}`~dorothy.data.normalizer.LabelNormalizer.transform_labels` |
| Predicted scatter | {py:func}`~dorothy.losses.heteroscedastic.predicted_scatter` |
| Model forward pass per label source | {py:func}`~dorothy.training.batches.forward_label_sources` |

`tests/test_end_to_end.py` checks that evaluation reproduces the training
validation metrics for every data layout, which catches a definition that
drifted between two callers.
