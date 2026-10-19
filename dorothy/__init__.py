"""DOROTHY: neural networks that infer stellar parameters and abundances from spectra.

The package trains networks on APOGEE and GALAH labels to predict stellar
parameters, abundances and their uncertainties from DESI, BOSS and LAMOST
(LRS and MRS) spectra.

Main components:
    - config: Configuration schema and loading utilities
    - data: Data loading, normalization, and preprocessing
    - models: Neural network architectures
    - losses: Loss functions including heteroscedastic loss
    - training: Training loop
    - inference: Prediction and evaluation
    - analysis: Saliency and k-NN anomaly detection for trained models, and
      label-source comparison
    - visualization: Plots of training and evaluation results
    - cli: Command-line interface
"""

__version__ = "0.1.0"
__author__ = "DOROTHY Team"
