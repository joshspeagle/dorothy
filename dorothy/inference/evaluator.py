"""
Evaluation metrics for DOROTHY stellar parameter predictions.

This module provides tools for computing per-parameter evaluation metrics
and comparing predictions against ground truth values.

Metrics computed:
    Mean-based (outlier-sensitive):
        - RMSE: Root Mean Square Error
        - Bias: Mean (predicted - true)
        - SD: Standard deviation of residuals

    Quantile-based (robust):
        - MAE: Mean Absolute Error
        - Median offset: Median (predicted - true)
        - Robust scatter: (P84 - P16) / 2 of residuals

    Z-score metrics (calibration check, when uncertainties provided):
        - z_median: Median of z-scores
        - z_robust_scatter: (P84 - P16) / 2 of z-scores (should be ~1 if calibrated)

    Predicted uncertainty statistics:
        - pred_unc_p16, pred_unc_p50, pred_unc_p84: Percentiles of predicted scatter

Example:
    >>> evaluator = Evaluator(parameter_names=["teff", "logg", "fe_h"])
    >>> result = evaluator.evaluate(y_pred, y_true, pred_scatter, label_errors)
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
        n_samples: Number of samples evaluated.

        # Mean-based metrics (outlier-sensitive)
        rmse: Root Mean Square Error.
        bias: Mean (predicted - true).
        sd: Standard deviation of residuals.

        # Quantile-based metrics (robust to outliers)
        mae: Mean Absolute Error.
        median_offset: Median (predicted - true).
        robust_scatter: (P84 - P16) / 2 of residuals.

        # Z-score metrics (calibration check)
        z_median: Median of z-scores (None if no uncertainties).
        z_robust_scatter: (P84 - P16) / 2 of z-scores (None if no uncertainties).

        # Predicted uncertainty statistics
        pred_unc_p16: 16th percentile of predicted scatter (None if no uncertainties).
        pred_unc_p50: 50th percentile of predicted scatter (None if no uncertainties).
        pred_unc_p84: 84th percentile of predicted scatter (None if no uncertainties).
    """

    name: str
    n_samples: int

    # Mean-based metrics
    rmse: float
    bias: float
    sd: float

    # Quantile-based metrics
    mae: float
    median_offset: float
    robust_scatter: float

    # Z-score metrics (optional)
    z_median: float | None = None
    z_robust_scatter: float | None = None

    # Predicted uncertainty statistics (optional)
    pred_unc_p16: float | None = None
    pred_unc_p50: float | None = None
    pred_unc_p84: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "rmse": self.rmse,
            "bias": self.bias,
            "sd": self.sd,
            "mae": self.mae,
            "median_offset": self.median_offset,
            "robust_scatter": self.robust_scatter,
            "z_median": self.z_median,
            "z_robust_scatter": self.z_robust_scatter,
            "pred_unc_p16": self.pred_unc_p16,
            "pred_unc_p50": self.pred_unc_p50,
            "pred_unc_p84": self.pred_unc_p84,
        }


@dataclass
class EvaluationResult:
    """Container for evaluation results across all parameters.

    Attributes:
        metrics: Dictionary mapping parameter names to their metrics.
        parameter_names: List of parameter names in order.
        survey_name: Optional name of the survey (for per-survey results).
    """

    metrics: dict[str, ParameterMetrics] = field(default_factory=dict)
    parameter_names: list[str] = field(default_factory=list)
    survey_name: str | None = None

    def __getitem__(self, param_name: str) -> ParameterMetrics:
        """Get metrics for a specific parameter."""
        return self.metrics[param_name]

    def summary(self, format: str = "text", include_zscore: bool = True) -> str:
        """Generate a summary of evaluation results.

        Args:
            format: Output format ("text" or "markdown").
            include_zscore: Whether to include z-score metrics if available.

        Returns:
            Formatted summary string.
        """
        if format == "markdown":
            return self._summary_markdown(include_zscore)
        return self._summary_text(include_zscore)

    def _summary_text(self, include_zscore: bool = True) -> str:
        """Generate plain text summary."""
        lines = ["Evaluation Results", "=" * 80]

        # Mean-based metrics
        lines.append("\nMean-based metrics (outlier-sensitive):")
        lines.append("-" * 80)
        header = f"{'Parameter':<12} {'RMSE':>12} {'Bias':>12} {'SD':>12}"
        lines.append(header)
        lines.append("-" * 80)

        for param_name in self.parameter_names:
            m = self.metrics[param_name]
            line = f"{m.name:<12} {m.rmse:>12.4f} {m.bias:>+12.4f} {m.sd:>12.4f}"
            lines.append(line)

        # Quantile-based metrics
        lines.append("\nQuantile-based metrics (robust):")
        lines.append("-" * 80)
        header = f"{'Parameter':<12} {'MAE':>12} {'Med.Offset':>12} {'Rob.Scatter':>12}"
        lines.append(header)
        lines.append("-" * 80)

        for param_name in self.parameter_names:
            m = self.metrics[param_name]
            line = f"{m.name:<12} {m.mae:>12.4f} {m.median_offset:>+12.4f} {m.robust_scatter:>12.4f}"
            lines.append(line)

        # Z-score metrics (if available)
        first_param = self.metrics[self.parameter_names[0]]
        if include_zscore and first_param.z_median is not None:
            lines.append("\nZ-score calibration (expect median~0, scatter~1):")
            lines.append("-" * 80)
            header = f"{'Parameter':<12} {'z_median':>12} {'z_scatter':>12} {'s_p16':>10} {'s_p50':>10} {'s_p84':>10}"
            lines.append(header)
            lines.append("-" * 80)

            for param_name in self.parameter_names:
                m = self.metrics[param_name]
                line = (
                    f"{m.name:<12} {m.z_median:>+12.4f} {m.z_robust_scatter:>12.4f} "
                    f"{m.pred_unc_p16:>10.4f} {m.pred_unc_p50:>10.4f} {m.pred_unc_p84:>10.4f}"
                )
                lines.append(line)

        lines.append("-" * 80)
        lines.append(f"Total samples: {first_param.n_samples}")

        return "\n".join(lines)

    def _summary_markdown(self, include_zscore: bool = True) -> str:
        """Generate markdown summary."""
        lines = ["## Evaluation Results", ""]

        # Mean-based metrics
        lines.append("### Mean-based metrics (outlier-sensitive)")
        lines.append("| Parameter | RMSE | Bias | SD |")
        lines.append("|-----------|------|------|-----|")

        for param_name in self.parameter_names:
            m = self.metrics[param_name]
            line = f"| {m.name} | {m.rmse:.4f} | {m.bias:+.4f} | {m.sd:.4f} |"
            lines.append(line)

        # Quantile-based metrics
        lines.append("")
        lines.append("### Quantile-based metrics (robust)")
        lines.append("| Parameter | MAE | Median Offset | Robust Scatter |")
        lines.append("|-----------|-----|---------------|----------------|")

        for param_name in self.parameter_names:
            m = self.metrics[param_name]
            line = f"| {m.name} | {m.mae:.4f} | {m.median_offset:+.4f} | {m.robust_scatter:.4f} |"
            lines.append(line)

        # Z-score metrics (if available)
        first_param = self.metrics[self.parameter_names[0]]
        if include_zscore and first_param.z_median is not None:
            lines.append("")
            lines.append("### Z-score calibration (expect median~0, scatter~1)")
            lines.append("| Parameter | z_median | z_scatter | s_p16 | s_p50 | s_p84 |")
            lines.append("|-----------|----------|-----------|-------|-------|-------|")

            for param_name in self.parameter_names:
                m = self.metrics[param_name]
                line = (
                    f"| {m.name} | {m.z_median:+.4f} | {m.z_robust_scatter:.4f} | "
                    f"{m.pred_unc_p16:.4f} | {m.pred_unc_p50:.4f} | {m.pred_unc_p84:.4f} |"
                )
                lines.append(line)

        lines.append("")
        lines.append(f"*Total samples: {first_param.n_samples}*")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Convert all metrics to a dictionary."""
        return {name: m.to_dict() for name, m in self.metrics.items()}


@dataclass
class SurveyEvaluationResult:
    """Container for evaluation results broken down by survey.

    Holds both overall metrics and per-survey metrics for multi-survey
    training scenarios.

    Attributes:
        overall: Overall metrics across all surveys.
        by_survey: Dictionary mapping survey names to their EvaluationResult.
        survey_names: List of survey names in order.
        survey_sample_counts: Number of samples per survey.
    """

    overall: EvaluationResult
    by_survey: dict[str, EvaluationResult] = field(default_factory=dict)
    survey_names: list[str] = field(default_factory=list)
    survey_sample_counts: dict[str, int] = field(default_factory=dict)

    def __getitem__(self, survey_name: str) -> EvaluationResult:
        """Get metrics for a specific survey."""
        return self.by_survey[survey_name]

    def summary(
        self, format: str = "text", include_zscore: bool = True, compact: bool = False
    ) -> str:
        """Generate a summary of per-survey evaluation results.

        Args:
            format: Output format ("text" or "markdown").
            include_zscore: Whether to include z-score metrics if available.
            compact: If True, show abbreviated per-survey table instead of full details.

        Returns:
            Formatted summary string.
        """
        if format == "markdown":
            return self._summary_markdown(include_zscore, compact)
        return self._summary_text(include_zscore, compact)

    def _summary_text(self, include_zscore: bool = True, compact: bool = False) -> str:
        """Generate plain text summary with per-survey breakdown."""
        lines = ["Multi-Survey Evaluation Results", "=" * 80]

        # Overall summary
        lines.append("\nOVERALL:")
        lines.append("-" * 80)
        first_param = self.overall.metrics[self.overall.parameter_names[0]]
        lines.append(f"Total samples: {first_param.n_samples}")

        # Survey sample counts
        lines.append("\nSamples per survey:")
        for survey in self.survey_names:
            count = self.survey_sample_counts.get(survey, 0)
            lines.append(f"  {survey}: {count:,}")

        # Overall metrics (compact table)
        lines.append("\nOverall RMSE by parameter:")
        header = "  " + " ".join(
            f"{p[:8]:>10}" for p in self.overall.parameter_names[:8]
        )
        lines.append(header)
        values = " ".join(
            f"{self.overall.metrics[p].rmse:>10.4f}"
            for p in self.overall.parameter_names[:8]
        )
        lines.append("  " + values)

        # Per-survey breakdown
        lines.append("\n" + "=" * 80)
        lines.append("PER-SURVEY BREAKDOWN:")

        if compact:
            # Compact view: one line per survey with key metrics
            lines.append("-" * 80)
            header = f"{'Survey':<15} {'N':>8} "
            header += " ".join(
                f"{'RMSE_' + p[:4]:>10}" for p in self.overall.parameter_names[:4]
            )
            lines.append(header)
            lines.append("-" * 80)

            for survey in self.survey_names:
                result = self.by_survey[survey]
                count = self.survey_sample_counts.get(survey, 0)
                line = f"{survey:<15} {count:>8} "
                line += " ".join(
                    f"{result.metrics[p].rmse:>10.4f}"
                    for p in self.overall.parameter_names[:4]
                )
                lines.append(line)
        else:
            # Full view: complete metrics per survey
            for survey in self.survey_names:
                lines.append(f"\n--- {survey.upper()} ---")
                result = self.by_survey[survey]
                # Add indented version of the survey's summary
                survey_summary = result._summary_text(include_zscore)
                for line in survey_summary.split("\n")[2:]:  # Skip header
                    lines.append("  " + line)

        lines.append("=" * 80)
        return "\n".join(lines)

    def _summary_markdown(
        self, include_zscore: bool = True, compact: bool = False
    ) -> str:
        """Generate markdown summary with per-survey breakdown."""
        lines = ["## Multi-Survey Evaluation Results", ""]

        # Survey sample counts
        lines.append("### Sample Counts")
        lines.append("| Survey | Samples |")
        lines.append("|--------|---------|")
        for survey in self.survey_names:
            count = self.survey_sample_counts.get(survey, 0)
            lines.append(f"| {survey} | {count:,} |")

        # Overall RMSE table
        lines.append("")
        lines.append("### Overall RMSE")
        header = "| " + " | ".join(self.overall.parameter_names) + " |"
        lines.append(header)
        sep = "|" + "|".join(["---"] * len(self.overall.parameter_names)) + "|"
        lines.append(sep)
        values = (
            "| "
            + " | ".join(
                f"{self.overall.metrics[p].rmse:.4f}"
                for p in self.overall.parameter_names
            )
            + " |"
        )
        lines.append(values)

        # Per-survey RMSE comparison table
        lines.append("")
        lines.append("### RMSE by Survey")
        header = "| Survey | " + " | ".join(self.overall.parameter_names[:6]) + " |"
        lines.append(header)
        sep = (
            "|--------|"
            + "|".join(["------"] * min(6, len(self.overall.parameter_names)))
            + "|"
        )
        lines.append(sep)

        for survey in self.survey_names:
            result = self.by_survey[survey]
            values = (
                f"| {survey} | "
                + " | ".join(
                    f"{result.metrics[p].rmse:.4f}"
                    for p in self.overall.parameter_names[:6]
                )
                + " |"
            )
            lines.append(values)

        if not compact:
            # Full per-survey details
            for survey in self.survey_names:
                lines.append("")
                lines.append(f"### {survey}")
                result = self.by_survey[survey]
                # Add the survey's full markdown summary
                survey_lines = result._summary_markdown(include_zscore).split("\n")
                lines.extend(survey_lines[2:])  # Skip header

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert all metrics to a dictionary."""
        return {
            "overall": self.overall.to_dict(),
            "by_survey": {name: r.to_dict() for name, r in self.by_survey.items()},
            "survey_names": self.survey_names,
            "survey_sample_counts": self.survey_sample_counts,
        }


class Evaluator:
    """Evaluator for stellar parameter predictions.

    Computes per-parameter metrics for comparing predictions against
    ground truth values. Handles special cases like Teff evaluation
    in both linear and log space.

    Attributes:
        parameter_names: Names of parameters to evaluate.
        teff_in_log: Whether to also compute Teff metrics in log10 space.
        scatter_floor: Minimum scatter value (for z-score computation).

    Example:
        >>> evaluator = Evaluator(["teff", "logg", "fe_h"])
        >>> result = evaluator.evaluate(predictions, ground_truth, pred_scatter, label_errors)
        >>> print(result.summary())
    """

    def __init__(
        self,
        parameter_names: list[str] | None = None,
        teff_in_log: bool = True,
        scatter_floor: float = 0.01,
    ) -> None:
        """Initialize the evaluator.

        Args:
            parameter_names: Names of stellar parameters. If None, uses
                default ["teff", "logg", "fe_h"].
            teff_in_log: Whether to also evaluate Teff in log10 space,
                adding a "log_teff" entry to the metrics.
            scatter_floor: Minimum scatter floor for z-score computation.
        """
        if parameter_names is None:
            parameter_names = ["teff", "logg", "fe_h"]

        self.parameter_names = parameter_names
        self.teff_in_log = teff_in_log
        self.scatter_floor = scatter_floor

    def evaluate(
        self,
        y_pred: NDArray[np.float32],
        y_true: NDArray[np.float32],
        pred_scatter: NDArray[np.float32] | None = None,
        label_errors: NDArray[np.float32] | None = None,
        mask: NDArray[np.float32] | None = None,
    ) -> EvaluationResult:
        """Evaluate predictions against ground truth.

        Args:
            y_pred: Predicted values of shape (n_samples, n_parameters).
            y_true: Ground truth values of shape (n_samples, n_parameters).
            pred_scatter: Predicted scatter (s) from the model, shape (n_samples, n_parameters).
            label_errors: Label measurement errors (sigma_label), shape (n_samples, n_parameters).
            mask: Optional mask of shape (n_samples, n_parameters). Values of 1 indicate
                valid labels, 0 indicates masked labels. When provided, only valid samples
                contribute to metrics for each parameter.

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

        # Validate uncertainty arrays if provided
        if pred_scatter is not None and pred_scatter.shape != y_pred.shape:
            raise ValueError(
                f"pred_scatter shape {pred_scatter.shape} doesn't match "
                f"y_pred shape {y_pred.shape}"
            )
        if label_errors is not None and label_errors.shape != y_pred.shape:
            raise ValueError(
                f"label_errors shape {label_errors.shape} doesn't match "
                f"y_pred shape {y_pred.shape}"
            )
        if mask is not None and mask.shape != y_pred.shape:
            raise ValueError(
                f"mask shape {mask.shape} doesn't match " f"y_pred shape {y_pred.shape}"
            )

        result = EvaluationResult(parameter_names=list(self.parameter_names))

        for i, param_name in enumerate(self.parameter_names):
            # Get per-parameter mask
            param_mask = mask[:, i].astype(bool) if mask is not None else None

            # Filter arrays based on mask
            if param_mask is not None:
                valid_idx = param_mask
                pred = y_pred[valid_idx, i]
                true = y_true[valid_idx, i]
                s = pred_scatter[valid_idx, i] if pred_scatter is not None else None
                sigma = label_errors[valid_idx, i] if label_errors is not None else None
            else:
                pred = y_pred[:, i]
                true = y_true[:, i]
                s = pred_scatter[:, i] if pred_scatter is not None else None
                sigma = label_errors[:, i] if label_errors is not None else None

            metrics = self._compute_metrics(param_name, pred, true, s, sigma)
            result.metrics[param_name] = metrics

            # Add log(Teff) metrics if requested
            if param_name.lower() == "teff" and self.teff_in_log:
                valid = (pred > 0) & (true > 0)
                if valid.sum() > 0:
                    log_pred = np.log10(pred[valid])
                    log_true = np.log10(true[valid])
                    # For log-space, uncertainties need to be transformed
                    # d(log10(x)) = dx / (x * ln(10))
                    log_s = None
                    log_sigma = None
                    if s is not None:
                        # Approximate: use mean pred for transformation
                        log_s = s[valid] / (pred[valid] * np.log(10))
                    if sigma is not None:
                        log_sigma = sigma[valid] / (true[valid] * np.log(10))

                    log_metrics = self._compute_metrics(
                        "log_teff", log_pred, log_true, log_s, log_sigma
                    )
                    result.metrics["log_teff"] = log_metrics
                    result.parameter_names.append("log_teff")

        return result

    def evaluate_by_survey(
        self,
        y_pred: NDArray[np.float32],
        y_true: NDArray[np.float32],
        survey_ids: NDArray[np.int32] | NDArray[np.str_] | list[str] | None = None,
        survey_names: list[str] | None = None,
        pred_scatter: NDArray[np.float32] | None = None,
        label_errors: NDArray[np.float32] | None = None,
        mask: NDArray[np.float32] | None = None,
        has_data: dict[str, NDArray[np.bool_]] | None = None,
    ) -> SurveyEvaluationResult:
        """Evaluate predictions with breakdown by survey.

        Computes overall metrics and per-survey metrics for multi-survey
        training scenarios. Supports two modes:

        1. **Exclusive mode** (survey_ids): Each sample belongs to exactly one survey.
           Use this when each spectrum comes from a single survey.

        2. **Non-exclusive mode** (has_data): Samples can belong to multiple surveys.
           Use this for multi-survey training where stars may have spectra from
           multiple surveys (e.g., both BOSS and DESI). Sample counts may sum to
           more than the total number of samples.

        Args:
            y_pred: Predicted values of shape (n_samples, n_parameters).
            y_true: Ground truth values of shape (n_samples, n_parameters).
            survey_ids: Survey membership for each sample (exclusive mode). Can be:
                - Integer array of survey indices (0, 1, 2, ...)
                - String array of survey names
                - List of survey names (one per sample)
                Ignored if has_data is provided.
            survey_names: List of survey names corresponding to survey indices.
                Required if survey_ids is integer array. If survey_ids is strings,
                this is inferred from unique values. Ignored if has_data is provided.
            pred_scatter: Predicted scatter (s) from the model.
            label_errors: Label measurement errors (sigma_label).
            mask: Optional mask of shape (n_samples, n_parameters).
            has_data: Dict mapping survey names to boolean arrays indicating which
                samples have data from that survey (non-exclusive mode). Takes
                precedence over survey_ids if both are provided.

        Returns:
            SurveyEvaluationResult with overall and per-survey metrics.

        Raises:
            ValueError: If neither survey_ids nor has_data is provided, or if
                input shapes don't match.

        Example (exclusive mode):
            >>> evaluator = Evaluator(["teff", "logg", "fe_h"])
            >>> survey_ids = np.array(["boss", "lamost", "boss", "lamost"])
            >>> result = evaluator.evaluate_by_survey(
            ...     y_pred, y_true, survey_ids,
            ...     pred_scatter=pred_scatter
            ... )
            >>> print(result.summary(compact=True))

        Example (non-exclusive mode):
            >>> has_data = {
            ...     "boss": np.array([True, True, False, True]),
            ...     "desi": np.array([False, True, True, True]),
            ... }
            >>> result = evaluator.evaluate_by_survey(
            ...     y_pred, y_true, has_data=has_data,
            ...     pred_scatter=pred_scatter
            ... )
        """
        # Determine which mode we're using
        if has_data is not None:
            return self._evaluate_by_survey_non_exclusive(
                y_pred, y_true, has_data, pred_scatter, label_errors, mask
            )
        elif survey_ids is not None:
            return self._evaluate_by_survey_exclusive(
                y_pred,
                y_true,
                survey_ids,
                survey_names,
                pred_scatter,
                label_errors,
                mask,
            )
        else:
            raise ValueError("Either survey_ids or has_data must be provided")

    def _evaluate_by_survey_exclusive(
        self,
        y_pred: NDArray[np.float32],
        y_true: NDArray[np.float32],
        survey_ids: NDArray[np.int32] | NDArray[np.str_] | list[str],
        survey_names: list[str] | None,
        pred_scatter: NDArray[np.float32] | None,
        label_errors: NDArray[np.float32] | None,
        mask: NDArray[np.float32] | None,
    ) -> SurveyEvaluationResult:
        """Evaluate with exclusive survey assignment (each sample in one survey)."""
        # Convert survey_ids to numpy array if needed
        if isinstance(survey_ids, list):
            survey_ids = np.array(survey_ids)

        # Determine survey names
        if survey_ids.dtype.kind in ("U", "S", "O"):  # String types
            # String array: extract unique names
            unique_surveys = list(np.unique(survey_ids))
            if survey_names is None:
                survey_names = unique_surveys
            # Convert to integer indices for consistency
            survey_name_to_idx = {name: i for i, name in enumerate(survey_names)}
            survey_indices = np.array(
                [survey_name_to_idx.get(s, -1) for s in survey_ids], dtype=np.int32
            )
        else:
            # Integer array: need survey_names
            if survey_names is None:
                # Create default names
                unique_indices = np.unique(survey_ids)
                survey_names = [f"survey_{i}" for i in unique_indices]
            survey_indices = survey_ids.astype(np.int32)

        # Compute overall metrics
        overall = self.evaluate(y_pred, y_true, pred_scatter, label_errors, mask=mask)

        # Compute per-survey metrics
        by_survey: dict[str, EvaluationResult] = {}
        survey_sample_counts: dict[str, int] = {}

        for survey_idx, survey_name in enumerate(survey_names):
            # Create boolean mask for this survey
            survey_mask = survey_indices == survey_idx
            n_survey_samples = survey_mask.sum()
            survey_sample_counts[survey_name] = int(n_survey_samples)

            if n_survey_samples == 0:
                # No samples for this survey - create empty result
                empty_result = EvaluationResult(
                    parameter_names=list(self.parameter_names),
                    survey_name=survey_name,
                )
                for param_name in self.parameter_names:
                    empty_result.metrics[param_name] = ParameterMetrics(
                        name=param_name,
                        n_samples=0,
                        rmse=float("nan"),
                        bias=float("nan"),
                        sd=float("nan"),
                        mae=float("nan"),
                        median_offset=float("nan"),
                        robust_scatter=float("nan"),
                    )
                by_survey[survey_name] = empty_result
                continue

            # Filter arrays for this survey
            survey_y_pred = y_pred[survey_mask]
            survey_y_true = y_true[survey_mask]
            survey_pred_scatter = (
                pred_scatter[survey_mask] if pred_scatter is not None else None
            )
            survey_label_errors = (
                label_errors[survey_mask] if label_errors is not None else None
            )
            survey_param_mask = mask[survey_mask] if mask is not None else None

            # Evaluate this survey
            survey_result = self.evaluate(
                survey_y_pred,
                survey_y_true,
                survey_pred_scatter,
                survey_label_errors,
                mask=survey_param_mask,
            )
            survey_result.survey_name = survey_name
            by_survey[survey_name] = survey_result

        return SurveyEvaluationResult(
            overall=overall,
            by_survey=by_survey,
            survey_names=survey_names,
            survey_sample_counts=survey_sample_counts,
        )

    def _evaluate_by_survey_non_exclusive(
        self,
        y_pred: NDArray[np.float32],
        y_true: NDArray[np.float32],
        has_data: dict[str, NDArray[np.bool_]],
        pred_scatter: NDArray[np.float32] | None,
        label_errors: NDArray[np.float32] | None,
        mask: NDArray[np.float32] | None,
    ) -> SurveyEvaluationResult:
        """Evaluate with non-exclusive survey assignment (samples can be in multiple surveys)."""
        survey_names = list(has_data.keys())

        # Compute overall metrics
        overall = self.evaluate(y_pred, y_true, pred_scatter, label_errors, mask=mask)

        # Compute per-survey metrics
        by_survey: dict[str, EvaluationResult] = {}
        survey_sample_counts: dict[str, int] = {}

        for survey_name in survey_names:
            survey_mask = has_data[survey_name]
            n_survey_samples = survey_mask.sum()
            survey_sample_counts[survey_name] = int(n_survey_samples)

            if n_survey_samples == 0:
                # No samples for this survey - create empty result
                empty_result = EvaluationResult(
                    parameter_names=list(self.parameter_names),
                    survey_name=survey_name,
                )
                for param_name in self.parameter_names:
                    empty_result.metrics[param_name] = ParameterMetrics(
                        name=param_name,
                        n_samples=0,
                        rmse=float("nan"),
                        bias=float("nan"),
                        sd=float("nan"),
                        mae=float("nan"),
                        median_offset=float("nan"),
                        robust_scatter=float("nan"),
                    )
                by_survey[survey_name] = empty_result
                continue

            # Filter arrays for this survey
            survey_y_pred = y_pred[survey_mask]
            survey_y_true = y_true[survey_mask]
            survey_pred_scatter = (
                pred_scatter[survey_mask] if pred_scatter is not None else None
            )
            survey_label_errors = (
                label_errors[survey_mask] if label_errors is not None else None
            )
            survey_param_mask = mask[survey_mask] if mask is not None else None

            # Evaluate this survey
            survey_result = self.evaluate(
                survey_y_pred,
                survey_y_true,
                survey_pred_scatter,
                survey_label_errors,
                mask=survey_param_mask,
            )
            survey_result.survey_name = survey_name
            by_survey[survey_name] = survey_result

        return SurveyEvaluationResult(
            overall=overall,
            by_survey=by_survey,
            survey_names=survey_names,
            survey_sample_counts=survey_sample_counts,
        )

    def _compute_metrics(
        self,
        name: str,
        pred: NDArray[np.float32],
        true: NDArray[np.float32],
        pred_scatter: NDArray[np.float32] | None = None,
        label_errors: NDArray[np.float32] | None = None,
    ) -> ParameterMetrics:
        """Compute metrics for a single parameter.

        Args:
            name: Parameter name.
            pred: Predicted values.
            true: Ground truth values.
            pred_scatter: Predicted scatter (s) from the model.
            label_errors: Label measurement errors (sigma_label).

        Returns:
            ParameterMetrics instance. If no valid samples, returns NaN for all metrics.
        """
        n_samples = len(pred)

        # Handle empty arrays (all samples masked)
        if n_samples == 0:
            return ParameterMetrics(
                name=name,
                n_samples=0,
                rmse=float("nan"),
                bias=float("nan"),
                sd=float("nan"),
                mae=float("nan"),
                median_offset=float("nan"),
                robust_scatter=float("nan"),
                z_median=float("nan") if pred_scatter is not None else None,
                z_robust_scatter=float("nan") if pred_scatter is not None else None,
                pred_unc_p16=float("nan") if pred_scatter is not None else None,
                pred_unc_p50=float("nan") if pred_scatter is not None else None,
                pred_unc_p84=float("nan") if pred_scatter is not None else None,
            )

        residuals = pred - true

        # Mean-based metrics (outlier-sensitive)
        rmse = np.sqrt(np.mean(residuals**2))
        bias = np.mean(residuals)
        sd = np.std(residuals)

        # Quantile-based metrics (robust)
        mae = np.mean(np.abs(residuals))
        median_offset = np.median(residuals)
        p16, p84 = np.percentile(residuals, [16, 84])
        robust_scatter = (p84 - p16) / 2

        # Z-score metrics (if uncertainties provided)
        z_median = None
        z_robust_scatter = None
        pred_unc_p16 = None
        pred_unc_p50 = None
        pred_unc_p84 = None

        if pred_scatter is not None:
            # Compute total variance: sigma_label^2 + s^2 + s_0^2
            s_squared = pred_scatter**2 + self.scatter_floor**2
            total_var = (
                label_errors**2 + s_squared if label_errors is not None else s_squared
            )

            total_std = np.sqrt(total_var)

            # Z-scores
            z_scores = residuals / total_std
            z_median = np.median(z_scores)
            z_p16, z_p84 = np.percentile(z_scores, [16, 84])
            z_robust_scatter = (z_p84 - z_p16) / 2

            # Predicted uncertainty statistics
            pred_unc_p16, pred_unc_p50, pred_unc_p84 = np.percentile(
                pred_scatter, [16, 50, 84]
            )

        return ParameterMetrics(
            name=name,
            n_samples=n_samples,
            rmse=float(rmse),
            bias=float(bias),
            sd=float(sd),
            mae=float(mae),
            median_offset=float(median_offset),
            robust_scatter=float(robust_scatter),
            z_median=float(z_median) if z_median is not None else None,
            z_robust_scatter=(
                float(z_robust_scatter) if z_robust_scatter is not None else None
            ),
            pred_unc_p16=float(pred_unc_p16) if pred_unc_p16 is not None else None,
            pred_unc_p50=float(pred_unc_p50) if pred_unc_p50 is not None else None,
            pred_unc_p84=float(pred_unc_p84) if pred_unc_p84 is not None else None,
        )


def evaluate_predictions(
    y_pred: NDArray[np.float32],
    y_true: NDArray[np.float32],
    pred_scatter: NDArray[np.float32] | None = None,
    label_errors: NDArray[np.float32] | None = None,
    parameter_names: list[str] | None = None,
    mask: NDArray[np.float32] | None = None,
) -> EvaluationResult:
    """Convenience function to evaluate predictions.

    Args:
        y_pred: Predicted values.
        y_true: Ground truth values.
        pred_scatter: Predicted scatter (s) from the model.
        label_errors: Label measurement errors (sigma_label).
        parameter_names: Optional list of parameter names.
        mask: Optional mask of shape (n_samples, n_parameters). Values of 1 indicate
            valid labels, 0 indicates masked labels.

    Returns:
        EvaluationResult with per-parameter metrics.
    """
    evaluator = Evaluator(parameter_names=parameter_names)
    return evaluator.evaluate(y_pred, y_true, pred_scatter, label_errors, mask=mask)


def evaluate_predictions_by_survey(
    y_pred: NDArray[np.float32],
    y_true: NDArray[np.float32],
    survey_ids: NDArray[np.int32] | NDArray[np.str_] | list[str] | None = None,
    survey_names: list[str] | None = None,
    pred_scatter: NDArray[np.float32] | None = None,
    label_errors: NDArray[np.float32] | None = None,
    parameter_names: list[str] | None = None,
    mask: NDArray[np.float32] | None = None,
    has_data: dict[str, NDArray[np.bool_]] | None = None,
) -> SurveyEvaluationResult:
    """Convenience function to evaluate predictions with per-survey breakdown.

    Supports two modes:
    1. **Exclusive mode** (survey_ids): Each sample belongs to exactly one survey.
    2. **Non-exclusive mode** (has_data): Samples can belong to multiple surveys.

    Args:
        y_pred: Predicted values.
        y_true: Ground truth values.
        survey_ids: Survey membership for each sample (integer indices or strings).
            Used for exclusive mode. Ignored if has_data is provided.
        survey_names: List of survey names (required if survey_ids is integers).
        pred_scatter: Predicted scatter (s) from the model.
        label_errors: Label measurement errors (sigma_label).
        parameter_names: Optional list of parameter names.
        mask: Optional mask of shape (n_samples, n_parameters).
        has_data: Dict mapping survey names to boolean arrays (non-exclusive mode).
            Takes precedence over survey_ids if both are provided.

    Returns:
        SurveyEvaluationResult with overall and per-survey metrics.
    """
    evaluator = Evaluator(parameter_names=parameter_names)
    return evaluator.evaluate_by_survey(
        y_pred,
        y_true,
        survey_ids,
        survey_names,
        pred_scatter,
        label_errors,
        mask=mask,
        has_data=has_data,
    )
