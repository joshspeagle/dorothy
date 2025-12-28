"""Analysis tools for DOROTHY models."""

from dorothy.analysis.knn_anomaly import (
    AnomalyDetector,
    AnomalyResult,
    l2_normalize,
)
from dorothy.analysis.saliency import (
    SaliencyAnalyzer,
    SaliencyResult,
    plot_parameter_saliency,
    plot_saliency_heatmap,
)


__all__ = [
    # Anomaly detection
    "AnomalyDetector",
    "AnomalyResult",
    "l2_normalize",
    # Saliency analysis
    "SaliencyAnalyzer",
    "SaliencyResult",
    "plot_parameter_saliency",
    "plot_saliency_heatmap",
]
