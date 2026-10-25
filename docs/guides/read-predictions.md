# Read the predictions

## Anomaly detection

{py:class}`~dorothy.analysis.knn_anomaly.AnomalyDetector` flags stars whose
embeddings are far from those of the training stars:

1. Compute the embeddings of the training stars and scale them to unit
   length.
2. For each star, find the distance to its k-th nearest training neighbor.
3. Flag the stars whose distance exceeds a percentile of the training
   stars' distances.

```python
from dorothy.analysis import AnomalyDetector

detector = AnomalyDetector.from_predictor(predictor)
detector.fit(X_train, k=10, percentile=99)
result = detector.detect(X_test)
result.get_anomaly_indices()
```

A flagged star has a spectrum unlike those the model was trained on, so its
predictions deserve less trust.
