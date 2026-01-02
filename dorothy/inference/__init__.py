"""Prediction and evaluation utilities."""

from dorothy.inference.evaluation_utils import (
    build_batch_from_sparse,
    evaluate_on_dense_data,
    evaluate_on_single_survey_data,
    evaluate_on_test_set,
    normalize_dense_data,
    normalize_single_survey_data,
    normalize_sparse_data,
)
from dorothy.inference.evaluator import (
    EvaluationResult,
    Evaluator,
    ParameterMetrics,
    SurveyEvaluationResult,
    evaluate_predictions,
    evaluate_predictions_by_survey,
)
from dorothy.inference.predictor import (
    PredictionResult,
    Predictor,
    predict_from_fits,
)


__all__ = [
    "Predictor",
    "PredictionResult",
    "predict_from_fits",
    "Evaluator",
    "EvaluationResult",
    "SurveyEvaluationResult",
    "ParameterMetrics",
    "evaluate_predictions",
    "evaluate_predictions_by_survey",
    "build_batch_from_sparse",
    "normalize_sparse_data",
    "normalize_dense_data",
    "normalize_single_survey_data",
    "evaluate_on_test_set",
    "evaluate_on_dense_data",
    "evaluate_on_single_survey_data",
]
