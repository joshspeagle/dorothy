# Overview

This page follows the data from the catalogue to a prediction. Each step
names the code that runs it and links to the page that explains it.

::::{container} d-figure
```{raw} html
:file: ../figures/pipeline.svg
:class: wide-only
```

```{raw} html
:file: ../figures/pipeline-narrow.svg
:class: narrow-only
```

:::{container} caption
The pipeline: training, evaluation and prediction share one implementation of
each step.
:::
::::

## The steps

1. **Select and split.**
   {py:func}`~dorothy.data.training_data.load_training_data` keeps the stars
   that have a spectrum from a configured survey and a valid label from a
   configured {term}`label source`. It splits them into training, validation
   and test stars ([Training](training.md#data-selection-and-split)).
2. **Preprocess.** Each spectrum becomes three channels: flux, the precision
   encoding ρ and a mask. Each label is scaled by the median and
   interquartile range of the training labels
   ([Preprocessing](preprocessing.md)).
3. **Predict.** The model maps the spectra of a star to a prediction μ and a
   log predicted scatter ln s for every label, one {term}`head` per label
   source ([The model](model.md)).
4. **Train.** The {py:class}`~dorothy.training.trainer.Trainer` minimizes the
   negative log-likelihood of the labels and keeps the weights of the epoch
   with the lowest validation loss ([Uncertainties](uncertainties.md#the-loss),
   [Training](training.md)).
5. **Calibrate.** After training, the trainer fits one
   {term}`calibration factor` per head and parameter, so that the validation
   z-scores have a robust spread of 1
   ([Uncertainties](uncertainties.md#calibration)).
6. **Save.** The {term}`checkpoint` holds the weights, the configuration, the
   label normalization, the calibration factors and the split
   ([Training](training.md#checkpoints)).
7. **Evaluate.** `dorothy evaluate` scores the test stars in normalized and
   physical units ([Evaluation](evaluation.md)).
8. **Predict new stars.** `dorothy predict` reads spectra from a FITS file and
   writes each label with its uncertainty
   ([Commands](../reference/commands.md#predict)).

## One definition per step

Training, evaluation and prediction call the same functions for the data
selection, the preprocessing, the label normalization, the predicted scatter,
its calibration and the forward pass. Evaluation on the validation stars
therefore reproduces the metrics recorded during training.
[Shared definitions](../contribute/shared-definitions.md) lists the functions.
