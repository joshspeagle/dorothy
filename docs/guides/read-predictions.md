# Read the predictions

This guide explains the columns of a prediction table and shows how to decide
which labels to trust.

## What you need

- A prediction table from `dorothy predict` ([Predict labels for your
  spectra](predict.md)).
- For Step 4, the checkpoint and the catalogue it was trained on.

## Step 1: Check the units

Each label column is in the units of the catalogue: K for `teff`, and dex for
`logg` and the abundances ([Labels and quality flags](../catalogue/labels.md)).
The column names and their order come from the catalogue's parameter names.
With `dorothy predict --normalized`, the columns are in
{term}`normalized units` instead ([Preprocessing](../methods/preprocessing.md)).

## Step 2: Read the uncertainty

The column `teff_err` holds the uncertainty of `teff`, and so on for each
label. It is the {term}`predicted scatter` s times the
{term}`calibration factor` c of that label source and parameter:

$$
\sigma_\mathrm{cal} = c\, s .
$$

s is the expected scatter between the prediction and the label source's own
measurement of the star, including that source's label noise. So the GALAH
head's uncertainty refers to GALAH labels, and the APOGEE head's to APOGEE
labels. [Uncertainties](../methods/uncertainties.md) derives s and c.

When `calibrated` is false, the uncertainty is s alone. This happens with
`--no-calibration`, with a checkpoint that has no `calibration.json`, and with
a model trained with `loss: mse`, whose s is untrained.

## Step 3: Check the calibration

```bash
dorothy evaluate outputs/boss_apogee/boss_apogee_final
```

For each parameter, `z_scatter` should be close to 1 and `z_median` close
to 0. If a parameter misses these on the test stars, its uncertainties have the
wrong size or its predictions have a bias ([Evaluate a model](evaluate.md)). The calibration factors hold for new stars
only when those stars resemble the validation stars.

## Step 4: Flag unusual spectra

{py:class}`~dorothy.analysis.knn_anomaly.AnomalyDetector` flags spectra that
are unlike the training spectra. Such a spectrum may come from a kind of star,
or a data problem, that the model never saw, so its labels deserve less trust.

First, load the predictor, the training spectra `X_train` and the spectra to
check, `X_test`. This code takes both from the catalogue, for a model with one
survey and one label source:

```python
from dorothy.data.training_data import DataSplit, load_training_data
from dorothy.inference import Predictor

checkpoint = "outputs/boss_apogee/boss_apogee_final"
predictor = Predictor.load(checkpoint)
data = load_training_data(predictor.config)
split = DataSplit.load(checkpoint, data)
X_train = data.X[split.train]  # shape (n_train, 3, n_wavelengths)
X_test = data.X[split.test]
```

Any spectra preprocessed as in training can take the place of `X_test`
([Use DOROTHY from Python](python.md)). For a multi-head model, load the
predictor with `survey=` and pass spectra of that survey.

Then fit the detector and flag the spectra:

```python
from dorothy.analysis import AnomalyDetector

detector = AnomalyDetector.from_predictor(predictor)
detector.fit(X_train, k=10, percentile=99)
result = detector.detect(X_test)
flagged = result.get_anomaly_indices()
```

The detector compares the model's embeddings of the
spectra, scaled to unit length ([The model](../methods/model.md)). For each
spectrum, it finds the distance to the k-th nearest training spectrum. It
flags the spectra whose distance exceeds the given percentile of the same
distance among the training spectra. With `percentile=99`, about 1 % of
spectra like the training spectra are flagged by chance.

## Step 5: Decide which labels to use

Trust a label when all of these hold:

1. `calibrated` is true, and `dorothy evaluate` gives that parameter a
   `z_scatter` close to 1.
2. The label source measures the parameter. A head still predicts a parameter
   that its label source lacks, but no label trained it.
3. The detector does not flag the spectrum.

## Next

- [Evaluate a model](evaluate.md): the metrics behind Step 3.
- [Inspect a model](inspect.md): which pixels a prediction depends on.
