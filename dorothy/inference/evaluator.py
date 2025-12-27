"""
Evaluation metrics for DOROTHY stellar parameter predictions.

This module provides tools for computing per-parameter evaluation metrics
and comparing predictions against ground truth values.

Metrics computed:
- MAE: Mean Absolute Error
- RMSE: Root Mean Square Error
- Bias: Mean (predicted - true)
- Scatter: Standard deviation of residuals
- Median Absolute Deviation (MAD)

Example:
    >>> evaluator = Evaluator(parameter_names=["teff", "logg", "feh"])
    >>> result = evaluator.evaluate(y_pred, y_true)
    >>> print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class ParameterMetrics:
    """Metrics for a single stellar parameter.

    Attributes:
        name: Parameter name.
        mae: Mean Absolute Error.
        rmse: Root Mean Square Error.
        bias: Mean (predicted - true).
        scatter: Standard deviation of residuals.
        mad: Median Absolute Deviation.
        n_samples: Number of samples evaluated.
    """

    name: str
    mae: float
    rmse: float
    bias: float
    scatter: float
    mad: float
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "mae": self.mae,
            "rmse": self.rmse,
            "bias": self.bias,
            "scatter": self.scatter,
            "mad": self.mad,
            "n_samples": self.n_samples,
        }


@dataclass
class EvaluationResult:
    """Container for evaluation results across all parameters.

    Attributes:
        metrics: Dictionary mapping parameter names to their metrics.
        parameter_names: List of parameter names in order.
    """

    metrics: dict[str, ParameterMetrics] = field(default_factory=dict)
    parameter_names: list[str] = field(default_factory=list)

    def __getitem__(self, param_name: str) -> ParameterMetrics:
        """Get metrics for a specific parameter."""
        return self.metrics[param_name]

    def summary(self, format: str = "text") -> str:
        """Generate a summary of evaluation results.

        Args:
            format: Output format ("text" or "markdown").

        Returns:
            Formatted summary string.
        """
        if format == "markdown":
            return self._summary_markdown()
        return self._summary_text()

    def _summary_text(self) -> str:
        """Generate plain text summary."""
        lines = ["Evaluation Results", "=" * 60]

        header = (
            f"{'Parameter':<10} {'MAE':>10} {'RMSE':>10} {'Bias':>10} {'Scatter':>10}"
        )
        lines.append(header)
        lines.append("-" * 60)

        for param_name in self.parameter_names:
            m = self.metrics[param_name]
            line = f"{m.name:<10} {m.mae:>10.4f} {m.rmse:>10.4f} {m.bias:>+10.4f} {m.scatter:>10.4f}"
            lines.append(line)

        lines.append("-" * 60)
        lines.append(
            f"Total samples: {self.metrics[self.parameter_names[0]].n_samples}"
        )

        return "\n".join(lines)

    def _summary_markdown(self) -> str:
        """Generate markdown summary."""
        lines = ["## Evaluation Results", ""]
        lines.append("| Parameter | MAE | RMSE | Bias | Scatter |")
        lines.append("|-----------|-----|------|------|---------|")

        for param_name in self.parameter_names:
            m = self.metrics[param_name]
            line = f"| {m.name} | {m.mae:.4f} | {m.rmse:.4f} | {m.bias:+.4f} | {m.scatter:.4f} |"
            lines.append(line)

        lines.append("")
        lines.append(
            f"*Total samples: {self.metrics[self.parameter_names[0]].n_samples}*"
        )

        return "\n".join(lines)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Convert all metrics to a dictionary."""
        return {name: m.to_dict() for name, m in self.metrics.items()}


class Evaluator:
    """Evaluator for stellar parameter predictions.

    Computes per-parameter metrics for comparing predictions against
    ground truth values. Handles special cases like Teff evaluation
    in both linear and log space.

    Attributes:
        parameter_names: Names of parameters to evaluate.
        teff_in_log: Whether to also compute Teff metrics in log10 space.

    Example:
        >>> evaluator = Evaluator(["teff", "logg", "feh"])
        >>> result = evaluator.evaluate(predictions, ground_truth)
        >>> print(result.summary())
    """

    def __init__(
        self,
        parameter_names: list[str] | None = None,
        teff_in_log: bool = True,
    ) -> None:
        """Initialize the evaluator.

        Args:
            parameter_names: Names of stellar parameters. If None, uses
                default ["teff", "logg", "feh"].
            teff_in_log: Whether to also evaluate Teff in log10 space,
                adding a "log_teff" entry to the metrics.
        """
        if parameter_names is None:
            parameter_names = ["teff", "logg", "feh"]

        self.parameter_names = parameter_names
        self.teff_in_log = teff_in_log

    def evaluate(
        self,
        y_pred: NDArray[np.float32],
        y_true: NDArray[np.float32],
        uncertainties: NDArray[np.float32] | None = None,
    ) -> EvaluationResult:
        """Evaluate predictions against ground truth.

        Args:
            y_pred: Predicted values of shape (n_samples, n_parameters).
            y_true: Ground truth values of shape (n_samples, n_parameters).
            uncertainties: Optional prediction uncertainties (unused currently).

        Returns:
            EvaluationResult containing per-parameter metrics.

        Raises:
            ValueError: If input shapes don't match.
        """
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}"
            )

        if y_pred.shape[1] != len(self.parameter_names):
            raise ValueError(
                f"Number of columns ({y_pred.shape[1]}) doesn't match "
                f"number of parameters ({len(self.parameter_names)})"
            )

        result = EvaluationResult(parameter_names=list(self.parameter_names))

        for i, param_name in enumerate(self.parameter_names):
            pred = y_pred[:, i]
            true = y_true[:, i]

            metrics = self._compute_metrics(param_name, pred, true)
            result.metrics[param_name] = metrics

            # Add log(Teff) metrics if requested
            if param_name.lower() == "teff" and self.teff_in_log:
                # Handle potential negative values (shouldn't happen with real data)
                valid = (pred > 0) & (true > 0)
                if valid.sum() > 0:
                    log_pred = np.log10(pred[valid])
                    log_true = np.log10(true[valid])
                    log_metrics = self._compute_metrics("log_teff", log_pred, log_true)
                    result.metrics["log_teff"] = log_metrics
                    result.parameter_names.append("log_teff")

        return result

    def _compute_metrics(
        self,
        name: str,
        pred: NDArray[np.float32],
        true: NDArray[np.float32],
    ) -> ParameterMetrics:
        """Compute metrics for a single parameter.

        Args:
            name: Parameter name.
            pred: Predicted values.
            true: Ground truth values.

        Returns:
            ParameterMetrics instance.
        """
        residuals = pred - true

        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals**2))
        bias = np.mean(residuals)
        scatter = np.std(residuals)
        mad = np.median(np.abs(residuals))

        return ParameterMetrics(
            name=name,
            mae=float(mae),
            rmse=float(rmse),
            bias=float(bias),
            scatter=float(scatter),
            mad=float(mad),
            n_samples=len(pred),
        )

    def evaluate_with_uncertainties(
        self,
        y_pred: NDArray[np.float32],
        y_true: NDArray[np.float32],
        y_true_err: NDArray[np.float32],
        pred_uncertainties: NDArray[np.float32],
    ) -> EvaluationResult:
        """Evaluate predictions accounting for uncertainties.

        Computes additional metrics like reduced chi-squared that account
        for both measurement and prediction uncertainties.

        Args:
            y_pred: Predicted values.
            y_true: Ground truth values.
            y_true_err: Ground truth measurement errors.
            pred_uncertainties: Prediction uncertainties from the model.

        Returns:
            EvaluationResult with additional uncertainty-aware metrics.
        """
        # For now, just compute basic metrics
        # Future: add chi-squared, calibration metrics, etc.
        return self.evaluate(y_pred, y_true, pred_uncertainties)


def evaluate_predictions(
    y_pred: NDArray[np.float32],
    y_true: NDArray[np.float32],
    parameter_names: list[str] | None = None,
) -> EvaluationResult:
    """Convenience function to evaluate predictions.

    Args:
        y_pred: Predicted values.
        y_true: Ground truth values.
        parameter_names: Optional list of parameter names.

    Returns:
        EvaluationResult with per-parameter metrics.
    """
    evaluator = Evaluator(parameter_names=parameter_names)
    return evaluator.evaluate(y_pred, y_true)
