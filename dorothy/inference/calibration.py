"""Calibration factors that scale the predicted scatter, one per head and parameter.

The trainer fits them after training (``training.calibrate_uncertainties``)
and saves them in the checkpoint as ``calibration.json``, which maps label
source -> parameter name -> factor. They describe ``best_model.pth``.
"""

from __future__ import annotations

import json
from pathlib import Path

CALIBRATION_FILE = "calibration.json"


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
