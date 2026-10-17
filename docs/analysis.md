# Analysis

`dorothy.analysis` has two tools for understanding a trained model: saliency
maps, which show the wavelengths that drive each prediction, and anomaly
detection in the model's latent space.

## Saliency maps

Both analyzers take one spectrum of shape `(3, n_wavelengths)`, preprocessed
as in training (see `CatalogueLoader.load_spectrum`), and its wavelength
grid. For a multi-survey model, `survey` selects the encoder.

### Gradient saliency

[`SaliencyAnalyzer`][dorothy.analysis.saliency.SaliencyAnalyzer] computes the
Jacobian of the predicted values and log-scatters with respect to the flux
and its uncertainty, and weights it by the Fisher information.

```python
from dorothy.analysis import SaliencyAnalyzer, plot_parameter_saliency

analyzer = SaliencyAnalyzer(model, parameter_names=names, scatter_floor=0.01)
result = analyzer.compute_saliency(X, wavelength, survey="boss")
plot_parameter_saliency(result, "teff", output_path="teff_saliency.png")
```

### Ablation saliency

[`AblationSaliencyAnalyzer`][dorothy.analysis.saliency.AblationSaliencyAnalyzer]
zeroes one block of pixels at a time, for every block position, and records
how the predictions and their uncertainties change. With
`use_training_distribution=True`, it averages over block sizes drawn as in
input masking during training ([Training](training.md#augmentation)).

```python
from dorothy.analysis import (
    AblationSaliencyAnalyzer,
    plot_ablation_parameter_saliency,
)

analyzer = AblationSaliencyAnalyzer(model, parameter_names=names)
result = analyzer.compute_saliency(
    X, wavelength, survey="boss", use_training_distribution=True, f_max=0.5
)
plot_ablation_parameter_saliency(result, "teff", output_path="teff_ablation.png")
```

`examples/saliency_example.py` and `examples/ablation_saliency_example.py`
load a checkpoint and a catalogue star and produce these maps.

## Anomaly detection

[`AnomalyDetector`][dorothy.analysis.knn_anomaly.AnomalyDetector] flags stars
whose latent vectors are far from the training stars:

1. Compute the embeddings of the training stars and normalize them to unit
   length.
2. For each star, find the distance to its k-th nearest training neighbor.
3. Flag stars whose distance exceeds a percentile of the training distances.

```python
from dorothy.analysis import AnomalyDetector

detector = AnomalyDetector.from_predictor(predictor)
detector.fit(X_train, k=10, percentile=99)
result = detector.detect(X_test)
result.get_anomaly_indices()
```
