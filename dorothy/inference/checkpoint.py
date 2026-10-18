"""
Loading trained DOROTHY checkpoints.

A checkpoint directory written by ``Trainer.save_checkpoint()`` contains the
model weights, ``config.yaml``, (usually) ``normalizer.pkl`` and the
calibration factors in ``calibration.json``, if training fit them. The model is
rebuilt from the saved config, so every architecture option (MLP or
MultiHeadMLP, normalization, activation, label sources, per-head dropout)
matches training. Directories with weights only fall back to inferring a
plain MLP, with an AdditiveOutput layer, from the state dict.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from dorothy.config.schema import ExperimentConfig
from dorothy.data.normalizer import LabelNormalizer
from dorothy.inference.calibration import load_calibration
from dorothy.models import MLP, MultiHeadMLP
from dorothy.utils import resolve_device

logger = logging.getLogger(__name__)


@dataclass
class LoadedCheckpoint:
    """A trained model with the artifacts needed to use it.

    Attributes:
        model: The model, in eval mode on the requested device.
        normalizer: Label normalizer, or None if the checkpoint has none.
        config: The experiment config the model was trained with, or None for
            weights-only checkpoints.
        path: The checkpoint directory.
        device: Device the model was moved to.
        calibration: Calibration factors of best_model.pth, label source ->
            parameter name -> factor, or None without calibration.json or
            for another weights file.
    """

    model: MLP | MultiHeadMLP
    normalizer: LabelNormalizer | None
    config: ExperimentConfig | None
    path: Path
    device: torch.device
    calibration: dict[str, dict[str, float]] | None

    @property
    def parameter_names(self) -> list[str]:
        """Names of the predicted parameters, in output order."""
        if self.normalizer is not None:
            return list(self.normalizer.parameters)
        if self.config is not None:
            return self.config.parameter_names
        from dorothy.config.schema import STELLAR_PARAMETERS

        n_params = self.model.n_parameters
        names = list(STELLAR_PARAMETERS[:n_params])
        return names + [f"param_{i}" for i in range(len(names), n_params)]

    @property
    def scatter_floor(self) -> float:
        """Scatter floor the model was trained with (default 0.01)."""
        if self.config is not None:
            return self.config.training.scatter_floor
        return 0.01


def _layer_sort_key(key: str) -> tuple:
    """Sort state-dict keys by the integers in them (layers.10 after layers.2)."""
    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", key))


def infer_mlp_architecture(state_dict: dict[str, torch.Tensor]) -> dict:
    """Infer MLP constructor arguments from a state dict.

    Recovers input/output sizes, hidden layer sizes (in layer order) and the
    normalization type. The activation cannot be inferred from weights and is
    left at the MLP default.

    Args:
        state_dict: MLP state dict.

    Returns:
        Dict of keyword arguments for :class:`~dorothy.models.MLP`.

    Raises:
        ValueError: If the state dict has no linear layers, or has a
            label-aware output layer (its config is needed to rebuild it).
    """
    if any(k.endswith("missing_log_error") for k in state_dict):
        raise ValueError(
            "state_dict has a label-aware output layer; load it from a "
            "checkpoint directory with its config.yaml"
        )
    linear = sorted(
        (k for k, v in state_dict.items() if k.endswith("weight") and v.dim() == 2),
        key=_layer_sort_key,
    )
    if not linear:
        raise ValueError("No linear layers found in state dict")
    normalization = (
        "batchnorm"
        if any(k.endswith("running_mean") for k in state_dict)
        else "layernorm"
    )
    return {
        "input_features": state_dict[linear[0]].shape[1],
        "output_features": state_dict[linear[-1]].shape[0],
        "hidden_layers": [state_dict[k].shape[0] for k in linear[:-1]],
        "normalization": normalization,
        "label_aware_uncertainty": False,
    }


def build_model(config: ExperimentConfig) -> MLP | MultiHeadMLP:
    """Build the (untrained) model described by an experiment config."""
    if config.is_multi_head:
        return MultiHeadMLP.from_config(config.multi_head_model)
    return MLP.from_config(config.model)


def head_label_sources(
    model: MLP | MultiHeadMLP, config: ExperimentConfig
) -> list[str]:
    """Label source of each output head, in model order.

    A single-head model's head is named after the configured label source.

    Args:
        model: The model.
        config: The experiment config the model was built from.

    Returns:
        One label source name per output head.
    """
    if isinstance(model, MultiHeadMLP) and model.label_sources != ["default"]:
        return list(model.label_sources)
    return list(config.data.label_sources[:1])


def load_checkpoint(
    checkpoint_path: str | Path,
    model_file: str = "best_model.pth",
    normalizer_file: str | None = "normalizer.pkl",
    device: str | torch.device = "auto",
) -> LoadedCheckpoint:
    """Load a trained model, its normalizer, config and calibration factors.

    Args:
        checkpoint_path: Checkpoint directory.
        model_file: Weights file inside the directory. The calibration
            factors in calibration.json describe best_model.pth, so other
            files load without them.
        normalizer_file: Normalizer file name (None to skip loading it).
        device: Device for the model ("auto", "cpu", "cuda", "mps").

    Returns:
        LoadedCheckpoint with the model in eval mode.

    Raises:
        FileNotFoundError: If the checkpoint or weights file does not exist.
        ValueError: If a checkpoint without config.yaml holds a model with
            label-aware uncertainty.
    """
    path = Path(checkpoint_path)
    model_path = path / model_file
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    device = resolve_device(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    config = None
    config_path = path / "config.yaml"
    if config_path.exists():
        config = ExperimentConfig.load_checkpoint_config(config_path)
        model = build_model(config)
    else:
        logger.warning(
            f"No config.yaml in {path}; inferring an MLP from the weights "
            "(activation assumed to be the default)"
        )
        model = MLP(**infer_mlp_architecture(state_dict))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    normalizer = None
    if normalizer_file is not None and (path / normalizer_file).exists():
        normalizer = LabelNormalizer.load(path / normalizer_file)

    return LoadedCheckpoint(
        model=model,
        normalizer=normalizer,
        config=config,
        path=path,
        device=device,
        calibration=(
            load_calibration(path)
            if config is not None and Path(model_file).name == "best_model.pth"
            else None
        ),
    )
