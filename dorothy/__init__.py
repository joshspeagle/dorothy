"""
DOROTHY: Deep learning framework for inferring stellar parameters from spectroscopic data.

This package provides tools for training neural networks to predict stellar atmospheric
parameters and chemical abundances from observed spectra across multiple astronomical
surveys (DESI, BOSS, LAMOST, APOGEE).

Main components:
    - config: Configuration schema and loading utilities
    - data: Data loading, normalization, and preprocessing
    - models: Neural network architectures
    - losses: Loss functions including heteroscedastic loss
    - training: Training loop and callbacks
    - inference: Prediction and evaluation
    - analysis: Post-hoc analysis tools (k-NN, saliency)
    - cli: Command-line interface
"""

__version__ = "0.1.0"
__author__ = "DOROTHY Team"
