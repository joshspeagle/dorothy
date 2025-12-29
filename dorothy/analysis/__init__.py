"""Analysis tools for DOROTHY models."""

from dorothy.analysis.knn_anomaly import (
    AnomalyDetector,
    AnomalyResult,
    l2_normalize,
)
from dorothy.analysis.saliency import (
    # Ablation-based saliency
    AblationSaliencyAnalyzer,
    AblationSaliencyResult,
    # Gradient-based saliency
    SaliencyAnalyzer,
    SaliencyResult,
    plot_ablation_parameter_saliency,
    plot_ablation_saliency_heatmap,
    plot_parameter_saliency,
    plot_saliency_heatmap,
)


__all__ = [
    # Anomaly detection
    "AnomalyDetector",
    "AnomalyResult",
    "l2_normalize",
    # Gradient-based saliency analysis
    "SaliencyAnalyzer",
    "SaliencyResult",
    "plot_parameter_saliency",
    "plot_saliency_heatmap",
    # Ablation-based saliency analysis
    "AblationSaliencyAnalyzer",
    "AblationSaliencyResult",
    "plot_ablation_parameter_saliency",
    "plot_ablation_saliency_heatmap",
]
