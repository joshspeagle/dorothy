"""Median/IQR label normalization for stellar parameters.

Labels are normalized to zero median and unit IQR (interquartile range). The
parameter named ``teff`` is normalized in log10 space. Error columns are
scaled by the same IQR as their labels.
"""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dorothy.config.schema import STELLAR_PARAMETERS


@dataclass
class ParameterStats:
    """
    Statistics for a single stellar parameter.

    Attributes:
        name: Parameter name (e.g., "teff", "logg", "fe_h").
        median: Median value for normalization.
        iqr: Interquartile range (75th - 25th percentile).
        use_log: Whether to apply log10 before normalization (for Teff).
        log_median: Median of log10 values (only if use_log=True).
        log_iqr: IQR of log10 values (only if use_log=True).

    Raises:
        ValueError: If ``use_log`` is set without ``log_median`` and
            ``log_iqr``.
    """

    name: str
    median: float
    iqr: float
    use_log: bool = False
    log_median: float | None = None
    log_iqr: float | None = None

    def __post_init__(self) -> None:
        """Validate that log stats are provided if use_log is True."""
        if self.use_log and (self.log_median is None or self.log_iqr is None):
            raise ValueError(
                f"log_median and log_iqr required when use_log=True for {self.name}"
            )


@dataclass
class LabelNormalizer:
    """
    Normalizer for stellar parameter labels using median/IQR scaling.

    The normalization is

        normalized = (value - median) / IQR

    and for the parameter named ``teff``

        normalized = (log10(value) - log_median) / log_IQR.

    Errors scale by the same IQR; the error of ``teff`` is first propagated to
    log10 space, sigma / (value ln 10).

    Attributes:
        parameters: List of parameter names to normalize.
        stats: Dictionary mapping parameter names to their statistics.
        is_fitted: Whether the normalizer has been fitted to data.

    Raises:
        ValueError: If ``parameters`` is empty.

    Examples:
        >>> normalizer = LabelNormalizer()
        >>> normalizer.fit(y_train)
        >>> y_train_norm = normalizer.transform(y_train)
        >>> y_pred_denorm = normalizer.inverse_transform(y_pred)
    """

    parameters: list[str] = field(default_factory=lambda: list(STELLAR_PARAMETERS))
    stats: dict[str, ParameterStats] = field(default_factory=dict)
    is_fitted: bool = False

    def __post_init__(self) -> None:
        """Validate parameter list."""
        if not self.parameters:
            raise ValueError("parameters list cannot be empty")

    @property
    def n_parameters(self) -> int:
        """Number of parameters being normalized."""
        return len(self.parameters)

    def fit(
        self,
        y: np.ndarray,
        errors: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> LabelNormalizer:
        """
        Fit the normalizer on training labels.

        Computes median and IQR for each parameter. For the parameter named
        ``teff``, also computes them in log10 space. A parameter with no
        unmasked value gets median 0 and IQR 1, with a warning.

        Args:
            y: Labels array of shape (n_samples, n_parameters).
            errors: Optional error array of shape (n_samples, n_parameters).
                Only its shape is checked.
            mask: Optional mask array of shape (n_samples, n_parameters).
                Labels are valid where > 0. Statistics use only valid
                values of each parameter.

        Returns:
            self, for method chaining.

        Raises:
            ValueError: If y is not 2-D with one column per parameter, errors
                or mask differ in shape from y, a valid value is NaN or
                infinite, a valid ``teff`` is not positive, or an IQR is zero.
        """
        if y.ndim != 2:
            raise ValueError(f"y must be 2D, got shape {y.shape}")

        if y.shape[1] != self.n_parameters:
            raise ValueError(
                f"y has {y.shape[1]} columns but expected {self.n_parameters} parameters"
            )

        if errors is not None and errors.shape != y.shape:
            raise ValueError(
                f"errors shape {errors.shape} doesn't match y shape {y.shape}"
            )

        if mask is not None and mask.shape != y.shape:
            raise ValueError(f"mask shape {mask.shape} doesn't match y shape {y.shape}")

        self.stats = {}

        for i, param_name in enumerate(self.parameters):
            # Get mask for this parameter if provided
            if mask is not None:
                param_mask = mask[:, i] > 0
                values = y[param_mask, i]
                if len(values) == 0:
                    # No valid labels (e.g. an element one label source lacks):
                    # use identity stats. The loss ignores these entries anyway.
                    warnings.warn(
                        f"All values masked for parameter {param_name}; "
                        "using identity normalization (median=0, IQR=1)",
                        UserWarning,
                        stacklevel=2,
                    )
                    self.stats[param_name] = ParameterStats(
                        name=param_name, median=0.0, iqr=1.0, use_log=False
                    )
                    continue
            else:
                values = y[:, i]

            # Check for invalid values in unmasked data
            if np.any(np.isnan(values)):
                raise ValueError(f"NaN values found in {param_name}")
            if np.any(np.isinf(values)):
                raise ValueError(f"Inf values found in {param_name}")

            # Teff uses log10 space (by convention, it's the first parameter)
            use_log = param_name == "teff"

            if use_log:
                if np.any(values <= 0):
                    raise ValueError(
                        f"Non-positive values found in {param_name} (log space)"
                    )

                log_values = np.log10(values)
                log_median = float(np.median(log_values))
                log_iqr = float(
                    np.percentile(log_values, 75) - np.percentile(log_values, 25)
                )

                if log_iqr == 0:
                    raise ValueError(f"Zero IQR in log space for {param_name}")

                self.stats[param_name] = ParameterStats(
                    name=param_name,
                    median=float(np.median(values)),
                    iqr=float(np.percentile(values, 75) - np.percentile(values, 25)),
                    use_log=True,
                    log_median=log_median,
                    log_iqr=log_iqr,
                )
            else:
                median = float(np.median(values))
                iqr = float(np.percentile(values, 75) - np.percentile(values, 25))

                if iqr == 0:
                    raise ValueError(f"Zero IQR for {param_name}")

                self.stats[param_name] = ParameterStats(
                    name=param_name,
                    median=median,
                    iqr=iqr,
                    use_log=False,
                )

        self.is_fitted = True
        return self

    def transform(
        self,
        y: np.ndarray,
        errors: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Normalize labels (and optionally errors).

        Args:
            y: Labels array of shape (n_samples, n_parameters).
            errors: Optional error array of shape (n_samples, n_parameters).
            mask: Optional mask array of shape (n_samples, n_parameters). Only
                its shape is checked; masked positions are normalized like the
                rest.

        Returns:
            If errors is None: float64 normalized labels of shape
            (n_samples, n_parameters). Otherwise a tuple of (normalized
            labels, normalized errors).

        Raises:
            RuntimeError: If normalizer has not been fitted.
            ValueError: If y is not 2-D with one column per parameter, or
                errors or mask differ in shape from y.
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer has not been fitted. Call fit() first.")

        if y.ndim != 2 or y.shape[1] != self.n_parameters:
            raise ValueError(
                f"y shape {y.shape} doesn't match expected (n_samples, {self.n_parameters})"
            )

        if errors is not None and errors.shape != y.shape:
            raise ValueError(
                f"errors shape {errors.shape} doesn't match y shape {y.shape}"
            )

        if mask is not None and mask.shape != y.shape:
            raise ValueError(f"mask shape {mask.shape} doesn't match y shape {y.shape}")

        y_norm = np.zeros_like(y, dtype=np.float64)
        errors_norm = np.zeros_like(y, dtype=np.float64) if errors is not None else None

        for i, param_name in enumerate(self.parameters):
            stats = self.stats[param_name]
            values = y[:, i]

            if stats.use_log:
                log_values = np.log10(values)
                y_norm[:, i] = (log_values - stats.log_median) / stats.log_iqr

                if errors is not None:
                    # Error propagation for log transform: d(log10(x))/dx = 1/(x*ln(10))
                    # So error in log space = error / (value * ln(10))
                    log_errors = errors[:, i] / (values * np.log(10))
                    errors_norm[:, i] = log_errors / stats.log_iqr
            else:
                y_norm[:, i] = (values - stats.median) / stats.iqr

                if errors is not None:
                    errors_norm[:, i] = errors[:, i] / stats.iqr

        if errors is not None:
            return y_norm, errors_norm
        return y_norm

    def transform_labels(self, y: np.ndarray) -> np.ndarray:
        """Normalize a 3-channel label array, leaving masked entries harmless.

        Masked entries (mask <= 0) are replaced by the parameter's training
        median (error 0) before transforming, so e.g. a masked Teff stored as
        0 cannot become log10(0) = -inf. The mask channel is unchanged.

        Args:
            y: Labels of shape (n_samples, 3, n_params) = [values, errors, mask].

        Returns:
            Normalized labels, same shape, float32.

        Raises:
            RuntimeError: If normalizer has not been fitted.
            ValueError: If y is not of shape (n_samples, 3, n_params).
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform")
        y = np.asarray(y)
        if y.ndim != 3 or y.shape[1] != 3:
            raise ValueError(
                f"y must have shape (n_samples, 3, n_params), got {y.shape}"
            )
        medians = np.array([self.stats[p].median for p in self.parameters])
        invalid = ~(y[:, 2, :] > 0)
        values = np.where(invalid, medians, y[:, 0, :].astype(np.float64))
        errors = np.where(invalid, 0.0, y[:, 1, :].astype(np.float64))
        values_norm, errors_norm = self.transform(values, errors)
        return np.stack([values_norm, errors_norm, y[:, 2, :]], axis=1).astype(
            np.float32
        )

    def inverse_transform(
        self,
        y_norm: np.ndarray,
        errors_norm: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Convert normalized predictions back to physical units.

        Errors of ``teff`` convert as sigma = value * ln 10 * sigma_log10.

        Args:
            y_norm: Normalized predictions of shape (n_samples, n_parameters).
            errors_norm: Optional normalized errors, same shape as y_norm.

        Returns:
            If errors_norm is None: float64 values of shape (n_samples,
            n_parameters). Otherwise a tuple of (values, errors).

        Raises:
            RuntimeError: If normalizer has not been fitted.
            ValueError: If y_norm is not 2-D with one column per parameter.
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer has not been fitted. Call fit() first.")

        if y_norm.ndim != 2 or y_norm.shape[1] != self.n_parameters:
            raise ValueError(
                f"y_norm shape {y_norm.shape} doesn't match expected "
                f"(n_samples, {self.n_parameters})"
            )

        y = np.zeros_like(y_norm, dtype=np.float64)
        errors = (
            np.zeros_like(y_norm, dtype=np.float64) if errors_norm is not None else None
        )

        for i, param_name in enumerate(self.parameters):
            stats = self.stats[param_name]
            values_norm = y_norm[:, i]

            if stats.use_log:
                # Inverse: log_value = normalized * log_iqr + log_median
                #          value = 10^log_value
                log_values = values_norm * stats.log_iqr + stats.log_median
                y[:, i] = 10**log_values

                if errors_norm is not None:
                    # Error in log space
                    log_errors = errors_norm[:, i] * stats.log_iqr
                    # Inverse of log transform error propagation
                    errors[:, i] = y[:, i] * np.log(10) * log_errors
            else:
                y[:, i] = values_norm * stats.iqr + stats.median

                if errors_norm is not None:
                    errors[:, i] = errors_norm[:, i] * stats.iqr

        if errors_norm is not None:
            return y, errors
        return y

    def get_params_dict(self) -> dict[str, dict[str, Any]]:
        """
        Get normalization parameters as a dictionary.

        Returns:
            Parameter name -> dict with ``median`` and ``IQR``, plus
            ``log_median`` and ``log_IQR`` for a log parameter. This is the
            format :meth:`save` writes.

        Raises:
            RuntimeError: If normalizer has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer has not been fitted. Call fit() first.")

        params = {}
        for param_name, stats in self.stats.items():
            params[param_name] = {
                "median": stats.median,
                "IQR": stats.iqr,
            }
            if stats.use_log:
                params[param_name]["log_median"] = stats.log_median
                params[param_name]["log_IQR"] = stats.log_iqr

        return params

    def save(self, path: str | Path) -> None:
        """
        Save normalizer parameters to a pickle file.

        Args:
            path: Path to save the pickle file.

        Raises:
            RuntimeError: If normalizer has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer has not been fitted. Call fit() first.")

        params = self.get_params_dict()
        with open(path, "wb") as f:
            pickle.dump(params, f)

    @classmethod
    def load(
        cls, path: str | Path, parameters: list[str] | None = None
    ) -> LabelNormalizer:
        """
        Load normalizer from a pickle file written by :meth:`save`.

        Args:
            path: Path to the pickle file. Only load files you trust.
            parameters: Optional list of parameter names. If not provided,
                uses the parameters found in the saved file.

        Returns:
            Fitted LabelNormalizer instance.

        Raises:
            ValueError: If a requested parameter is not in the file.
        """
        with open(path, "rb") as f:
            params = pickle.load(f)

        if parameters is None:
            # Use parameters from saved file
            parameters = list(params.keys())

        normalizer = cls(parameters=parameters)

        for param_name in parameters:
            if param_name not in params:
                raise ValueError(f"Parameter {param_name} not found in saved params")

            param_dict = params[param_name]
            use_log = "log_median" in param_dict or "log_IQR" in param_dict

            normalizer.stats[param_name] = ParameterStats(
                name=param_name,
                median=param_dict["median"],
                iqr=param_dict["IQR"],
                use_log=use_log,
                log_median=param_dict.get("log_median"),
                log_iqr=param_dict.get("log_IQR"),
            )

        normalizer.is_fitted = True
        return normalizer
