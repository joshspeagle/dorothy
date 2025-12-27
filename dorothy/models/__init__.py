"""Neural network architectures for stellar parameter inference."""

from dorothy.models.mlp import MLP
from dorothy.models.multi_head_mlp import (
    MultiHeadMLP,
    OutputHead,
    SharedTrunk,
    SurveyEncoder,
)


__all__ = [
    "MLP",
    "MultiHeadMLP",
    "SurveyEncoder",
    "SharedTrunk",
    "OutputHead",
]
