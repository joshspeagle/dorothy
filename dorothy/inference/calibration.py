"""Calibration factors that scale the predicted scatter, one per head and parameter.

The trainer fits them after training (``training.calibrate_uncertainties``)
and saves them in the checkpoint as ``calibration.json``, which maps label
source -> parameter name -> factor. They describe ``best_model.pth``.
Prediction and evaluation apply them with :func:`calibrate_scatter`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CALIBRATION_FILE = "calibration.json"


def calibrate_scatter(
    scatter: np.ndarray, factors: dict[str, float], parameter_names: list[str]
) -> np.ndarray:
    """Scale one head's predicted scatter by its calibration factors.

    sigma_cal = c * s, with the factor c of each parameter.

    Args:
        scatter: Predicted scatter s, shape (n_stars, n_params).
        factors: Parameter name -> calibration factor c.
        parameter_names: Parameter name of each column of scatter.

    Returns:
        The calibrated scatter, a new array with the shape and dtype of
        scatter.

    Raises:
        ValueError: If factors does not hold one factor for each name in
            parameter_names.
    """
    if set(factors) != set(parameter_names):
        raise ValueError(
            f"factors must hold one factor for each parameter in "
            f"{parameter_names}, got {sorted(factors)}"
        )
    return scatter * np.array(
        [factors[p] for p in parameter_names], dtype=scatter.dtype
    )


def save_calibration(
    checkpoint_path: str | Path, calibration: dict[str, dict[str, float]]
) -> None:
    """Write calibration factors to ``calibration.json`` in a checkpoint directory.

    Args:
        checkpoint_path: Checkpoint directory.
        calibration: Label source -> parameter name -> factor.
    """
    with open(Path(checkpoint_path) / CALIBRATION_FILE, "w") as f:
        json.dump(calibration, f, indent=2)


def load_calibration(checkpoint_path: str | Path) -> dict[str, dict[str, float]] | None:
    """Read the calibration factors of a checkpoint directory.

    Args:
        checkpoint_path: Checkpoint directory.

    Returns:
        Label source -> parameter name -> factor, or None if the directory
        has no calibration.json.
    """
    path = Path(checkpoint_path) / CALIBRATION_FILE
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
