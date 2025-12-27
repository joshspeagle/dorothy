# CLAUDE.md - AI Assistant Guide for DOROTHY

## Project Overview

Deep learning framework for inferring stellar parameters from spectroscopic data across multiple astronomical surveys (DESI, BOSS, LAMOST, APOGEE). This is a refactored, modular Python package replacing the original notebook-based experiments.

## Repository Status

| Metric | Value |
|--------|-------|
| Package Version | 0.1.0 |
| Python | >=3.10 |
| Tests | 233 passing |
| Refactoring Stage | Phase 3 (Analysis & Integration) - Complete |

## Current Progress

### Phase 1 - Foundation (Complete)

| Module | Path | Tests | Description |
|--------|------|-------|-------------|
| Config Schema | `dorothy/config/schema.py` | 29 | Pydantic models for experiment configuration |
| Heteroscedastic Loss | `dorothy/losses/heteroscedastic.py` | 20 | Uncertainty-aware loss function |
| MLP Architecture | `dorothy/models/mlp.py` | 26 | Configurable neural network |
| Label Normalizer | `dorothy/data/normalizer.py` | 29 | Median/IQR normalization with Teff log-space |
| Trainer | `dorothy/training/trainer.py` | 23 | Training loop with scheduling, checkpointing |

### Phase 2 - Data Pipeline & Inference (Complete)

| Module | Path | Tests | Description |
|--------|------|-------|-------------|
| FITS Loader | `dorothy/data/fits_loader.py` | 32 | Load FITS files, normalize spectra, quality filtering |
| Predictor | `dorothy/inference/predictor.py` | 19 | Model loading, batch prediction, denormalization |
| CLI | `dorothy/cli/main.py` | 21 | Command-line interface (train, predict, info) |

### Phase 3 - Analysis & Integration (Complete)

| Module | Path | Tests | Description |
|--------|------|-------|-------------|
| k-NN Anomaly Detection | `dorothy/analysis/knn_anomaly.py` | 21 | Embedding-based anomaly detection |
| Integration Tests | `tests/test_integration.py` | 13 | End-to-end workflow tests |
| Example Configs | `examples/*.yaml` | - | Example YAML configurations |

### Future (Not Started)

| Module | Path | Status |
|--------|------|--------|
| Saliency Maps | `dorothy/analysis/saliency.py` | Not started |
| Survey-specific Loaders | `dorothy/data/surveys/` | Not started |

Note: `dorothy/inference/evaluator.py` exists as a placeholder.

## Package Structure

```
dorothy/
├── pyproject.toml              # Package configuration
├── CLAUDE.md                   # This file
├── README.md                   # User documentation
│
├── dorothy/                    # Main package
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── schema.py           # Pydantic configuration models
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── normalizer.py       # Label normalization
│   │   └── fits_loader.py      # FITS file loading and preprocessing
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── mlp.py              # MLP architecture
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   └── heteroscedastic.py  # Loss functions
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py          # Training loop and checkpointing
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor.py        # Model loading and prediction
│   │   └── evaluator.py        # Metrics evaluation (placeholder)
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── knn_anomaly.py      # k-NN anomaly detection
│   │
│   └── cli/
│       ├── __init__.py
│       └── main.py             # Command-line interface
│
├── tests/
│   ├── test_cli.py
│   ├── test_config.py
│   ├── test_evaluator.py
│   ├── test_fits_loader.py
│   ├── test_integration.py     # End-to-end integration tests
│   ├── test_knn_anomaly.py
│   ├── test_losses.py
│   ├── test_models.py
│   ├── test_normalizer.py
│   ├── test_predictor.py
│   └── test_trainer.py
│
├── examples/                   # Example configurations
│   ├── basic_training.yaml     # Basic training config
│   └── advanced_training.yaml  # Config with masking
│
└── d5martin/                   # Original notebooks (reference only)
```

## Quick Commands

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_models.py -v

# Install in development mode
pip install -e ".[dev]"

# Format code
black dorothy/ tests/

# Lint code
ruff check dorothy/ tests/

# CLI commands
dorothy train config.yaml                    # Train a model
dorothy predict --checkpoint ./model --input data.fits --output predictions.csv
dorothy info ./model                         # Show checkpoint info
```

## Architecture

### Model Architecture (MLP)

```
Input: 2 channels × 7650 wavelengths = 15,300 features
       (or 4506 for BOSS multi-arm spectra)

MLP: 15300 → 5000 → 2000 → 1000 → 500 → 200 → 100 → 22
     (with BatchNorm/LayerNorm + GELU + optional Dropout)

Output: 11 stellar parameters + 11 uncertainties
```

### Stellar Parameters Predicted

1. **Atmospheric**: Teff (log-space), log g, [Fe/H]
2. **Abundances**: [Mg/Fe], [C/Fe], [Si/Fe], [Ni/Fe], [Al/Fe], [Ca/Fe], [N/Fe], [Mn/Fe]

### Loss Function (Heteroscedastic)

```python
s = sqrt(exp(2 * ln_s) + s_0²)           # Model scatter with floor
loss = (μ - y)² / (σ_label² + s²)        # Weighted squared error
     + log(σ_label² + s²)                # Log-variance penalty
```

Where:
- `μ`: predicted mean
- `ln_s`: predicted log-scatter
- `y`: true label
- `σ_label`: measurement uncertainty
- `s_0`: scatter floor (default 0.01)

### Normalization

- **Spectra**: `(flux - median) / IQR` per star
- **Labels**: `(value - median) / IQR` per parameter
- **Teff**: Uses log10 space before normalization

## Key Conventions

| Setting | Value |
|---------|-------|
| Random Seed | 42 |
| Train/Val/Test Split | 70% / 20% / 10% |
| Gradient Clipping | max_norm=10 |
| Default Epochs | 300 |
| Default Batch Size | 1024 |
| Optimizer | Adam (lr=1e-3) |
| Scheduler | CyclicLR (exp_range) |

## Configuration Example

```python
from dorothy.config import ExperimentConfig, DataConfig, ModelConfig
from pathlib import Path

config = ExperimentConfig(
    name="my_experiment",
    data=DataConfig(
        fits_path=Path("/data/spectra.fits"),
        survey="desi",
        wavelength_bins=7650,
    ),
    model=ModelConfig(
        hidden_layers=[5000, 2000, 1000, 500, 200, 100],
        normalization="batchnorm",
        activation="gelu",
    ),
)
```

## Important Notes for AI Assistants

1. **Always run tests** after making changes: `pytest tests/ -v`
2. **Teff uses log-space normalization** - different from other parameters
3. **BatchNorm requires batch_size > 1** during training
4. **Model outputs are normalized** - use LabelNormalizer.inverse_transform()
5. **Uncertainty predictions are log-scale** - apply exp() via loss.get_predicted_scatter()

## Reference Files (d5martin/)

The `d5martin/` folder contains the original notebooks for reference. **Note**: This folder is gitignored and not tracked in the repository - it exists only locally for reference.

| File | Purpose |
|------|---------|
| `DOROTHY_training.ipynb` | Complete training example |
| `DOROTHY/predict.py` | Chunk-based inference |
| `DOROTHY/run_knn.py` | k-NN anomaly detection |

These should not be modified - they are kept for reference during refactoring.

## CI/CD and Code Quality

- **Pre-commit hooks**: Black formatting and Ruff linting run automatically before each commit
- **GitHub Actions**: CI workflow runs linting and tests on push/PR to main
- **Python versions tested**: 3.10, 3.11, 3.12

Setup pre-commit locally:

```bash
pip install pre-commit
pre-commit install
```

## Development Workflow

1. Create/modify module in `dorothy/`
2. Write tests in `tests/test_<module>.py`
3. Run tests: `pytest tests/ -v`
4. Format: `black dorothy/ tests/`
5. Lint: `ruff check dorothy/ tests/`
