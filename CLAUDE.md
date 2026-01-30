# CLAUDE.md - AI Assistant Guide for DOROTHY

## Project Overview

Deep learning framework for inferring stellar parameters from spectroscopic data across multiple astronomical surveys (DESI, BOSS, LAMOST, APOGEE). This is a production-ready, modular Python package with support for multi-survey training and multi-label source configurations.

## Repository Status

| Metric | Value |
|--------|-------|
| Package Version | 0.1.0 |
| Python | >=3.10 |
| Tests | 653 passing |
| Test Coverage | 88% overall |
| Refactoring Stage | Complete |

## Module Overview

### Core Modules

| Module | Path | Coverage | Description |
|--------|------|----------|-------------|
| Config Schema | `dorothy/config/schema.py` | 93% | Pydantic models for experiment configuration |
| Heteroscedastic Loss | `dorothy/losses/heteroscedastic.py` | 97% | Uncertainty-aware loss function |
| MLP Architecture | `dorothy/models/mlp.py` | 100% | Standard configurable neural network |
| Multi-Head MLP | `dorothy/models/multi_head_mlp.py` | 91% | Multi-survey/multi-label architecture |
| Label Normalizer | `dorothy/data/normalizer.py` | 94% | Median/IQR normalization with Teff log-space |
| Trainer | `dorothy/training/trainer.py` | 91% | Training loop with multi-survey/multi-label support |

### Data Pipeline

| Module | Path | Coverage | Description |
|--------|------|----------|-------------|
| FITS Loader | `dorothy/data/fits_loader.py` | 94% | Load FITS files, normalize spectra |
| Catalogue Loader | `dorothy/data/catalogue_loader.py` | 79% | Multi-survey HDF5 super-catalogue |
| Augmentation | `dorothy/data/augmentation.py` | 97% | Dynamic input/label masking for training robustness |

### Inference & Analysis

| Module | Path | Coverage | Description |
|--------|------|----------|-------------|
| Predictor | `dorothy/inference/predictor.py` | 87% | Model loading, batch prediction |
| Evaluator | `dorothy/inference/evaluator.py` | 89% | Comprehensive metrics (RMSE, MAE, z-scores) |
| Evaluation Utils | `dorothy/inference/evaluation_utils.py` | 95% | Shared evaluation pipeline matching training validation |
| k-NN Anomaly | `dorothy/analysis/knn_anomaly.py` | 100% | Embedding-based anomaly detection |
| Saliency Analysis | `dorothy/analysis/saliency.py` | 95% | Gradient and ablation-based saliency maps for interpretability |

### User Interface

| Module | Path | Coverage | Description |
|--------|------|----------|-------------|
| CLI | `dorothy/cli/main.py` | 85% | Command-line interface |
| Training Plots | `dorothy/visualization/training_plots.py` | 90% | Training report generation |

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
│   │   ├── fits_loader.py      # FITS file loading and preprocessing
│   │   ├── catalogue_loader.py # Multi-survey HDF5 loading
│   │   └── augmentation.py     # Dynamic block masking
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mlp.py              # Standard MLP architecture
│   │   ├── multi_head_mlp.py   # Multi-survey/multi-label architecture
│   │   └── utils.py            # Shared model utilities
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   └── heteroscedastic.py  # Uncertainty-aware loss
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py          # Training with multi-survey/label support
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor.py        # Model loading and prediction
│   │   ├── evaluator.py        # Comprehensive metrics evaluation
│   │   └── evaluation_utils.py # Shared evaluation pipeline (matches training)
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── knn_anomaly.py      # k-NN anomaly detection
│   │   └── saliency.py         # Gradient and ablation-based saliency analysis
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── training_plots.py   # Training report generation
│   │
│   └── cli/
│       ├── __init__.py
│       └── main.py             # Command-line interface
│
├── scripts/                    # Catalogue building scripts
│   ├── build_catalogue_v2.py   # Main catalogue builder (v2)
│   ├── create_deduplicated_catalogue.py
│   └── legacy/                 # Deprecated v1 scripts
│       ├── add_boss_to_hdf5.py
│       ├── add_desi_to_hdf5.py
│       ├── add_lamost_lrs_to_hdf5.py
│       ├── add_lamost_mrs_to_hdf5.py
│       ├── add_gaia_ids_to_hdf5.py
│       └── fix_wavelengths.py
│
├── tests/                      # Test suite (653 tests)
│   ├── test_augmentation.py
│   ├── test_catalogue_loader.py
│   ├── test_cli.py
│   ├── test_config.py
│   ├── test_evaluator.py
│   ├── test_fits_loader.py
│   ├── test_integration.py
│   ├── test_knn_anomaly.py
│   ├── test_losses.py
│   ├── test_models.py
│   ├── test_multi_head_mlp.py
│   ├── test_normalizer.py
│   ├── test_predictor.py
│   ├── test_saliency.py
│   ├── test_trainer.py
│   └── test_visualization.py
│
├── examples/                   # Example configurations
│   ├── boss_training.yaml            # Quick BOSS test
│   ├── variant1_boss_apogee.yaml     # Single survey
│   ├── variant2_multi_survey.yaml    # Multi-survey training
│   ├── variant3_multi_labelset.yaml  # Multi-label heads
│   ├── variant4_all_surveys.yaml     # All surveys, single label
│   ├── variant5_all_surveys_masked.yaml  # All surveys with masking
│   ├── saliency_example.py           # Gradient saliency analysis demo
│   └── ablation_saliency_example.py  # Ablation saliency analysis demo
│
├── docs/
│   ├── super_catalogue.md      # HDF5 catalogue documentation
│   └── line_spread_functions.md # Survey LSF/resolution specifications
│
└── d5martin/                   # Original notebooks (reference, gitignored)
```

## Quick Commands

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=dorothy --cov-report=html

# Install in development mode
pip install -e ".[dev]"

# Format and lint
black dorothy/ tests/
ruff check dorothy/ tests/

# CLI commands
dorothy train config.yaml                    # Train a model
dorothy predict --checkpoint ./model --input data.fits --output predictions.csv
dorothy info ./model                         # Show checkpoint info
dorothy evaluate ./model                     # Evaluate on held-out test set
dorothy evaluate ./model --output results.json --format json  # Save results
```

## Architecture

### Standard MLP (Single-Survey)

```
Input: 2-3 channels × N wavelengths (survey-dependent)
       DESI: 7650, BOSS: 4506, LAMOST LRS: 3473

MLP: input_dim → 5000 → 2000 → 1000 → 500 → 200 → 100 → 22
     (with BatchNorm/LayerNorm + GELU + optional Dropout)

Output: 11 stellar parameters + 11 uncertainties
```

### Multi-Head MLP (Multi-Survey/Multi-Label)

```
           ┌─────────────────┐
 BOSS ────►│ SurveyEncoder   │───┐
           │ (4506 → 256)    │   │
           └─────────────────┘   │    ┌──────────────────┐
                                 ├───►│   SharedTrunk    │
           ┌─────────────────┐   │    │ (concat/mean)    │
 DESI ────►│ SurveyEncoder   │───┘    │ [512, 256] → 256 │
           │ (7650 → 256)    │        └────────┬─────────┘
           └─────────────────┘                 │
                                 ┌─────────────┴─────────────┐
                                 │                           │
                          ┌──────▼───────┐           ┌───────▼──────┐
                          │ OutputHead   │           │ OutputHead   │
                          │ (APOGEE)     │           │ (GALAH)      │
                          │ 256 → 22     │           │ 256 → 22     │
                          └──────────────┘           └──────────────┘
```

Components:
- **SurveyEncoder**: Per-survey input encoding (handles different wavelength grids)
- **SharedTrunk**: Combines encodings via concatenation or mean
- **OutputHead**: Per-labelset prediction head (one per label source)

### Stellar Parameters

| Index | Parameter | Description | Units |
|-------|-----------|-------------|-------|
| 0 | teff | Effective temperature | K (log-space) |
| 1 | logg | Surface gravity | log(cm/s²) |
| 2 | fe_h | Iron abundance | [Fe/H] dex |
| 3 | mg_fe | Magnesium | [Mg/Fe] dex |
| 4 | c_fe | Carbon | [C/Fe] dex |
| 5 | si_fe | Silicon | [Si/Fe] dex |
| 6 | ni_fe | Nickel | [Ni/Fe] dex |
| 7 | al_fe | Aluminum | [Al/Fe] dex |
| 8 | ca_fe | Calcium | [Ca/Fe] dex |
| 9 | n_fe | Nitrogen | [N/Fe] dex |
| 10 | mn_fe | Manganese | [Mn/Fe] dex |

### Loss Function (Heteroscedastic)

```python
s = sqrt(exp(2 * ln_s) + s_0²)           # Model scatter with floor
loss = (μ - y)² / (σ_label² + s²)        # Mean component (weighted squared error)
     + log(σ_label² + s²)                # Scatter component (log-variance penalty)
```

Where:
- `μ`: predicted mean
- `ln_s`: predicted log-scatter
- `y`: true label
- `σ_label`: measurement uncertainty
- `s_0`: scatter floor (default 0.01)

## Training Configurations

### Variant 1: Single Survey (Baseline)

```yaml
data:
  catalogue_path: data/super_catalogue.h5
  survey: boss
  label_source: apogee
```

### Variant 2: Multi-Survey (Shared Labels)

```yaml
data:
  catalogue_path: data/super_catalogue.h5
  surveys: [boss, desi]
  label_source: apogee

model:
  architecture: multi_head
  latent_dim: 256
  combination_mode: concat
```

### Variant 3: Multi-Survey + Multi-Label

```yaml
data:
  catalogue_path: data/super_catalogue.h5
  surveys: [boss, desi]
  label_sources: [apogee, galah]

model:
  architecture: multi_head
  latent_dim: 256
```

## Key Conventions

| Setting | Value |
|---------|-------|
| Random Seed | 42 |
| Train/Val/Test Split | 70% / 20% / 10% |
| Gradient Clipping | max_norm=10 |
| Default Epochs | 100-300 |
| Default Batch Size | 512-1024 |
| Optimizer | AdamW (lr=1e-3, weight_decay=0.01) |
| Scheduler | OneCycleLR |

## Important Notes for AI Assistants

1. **Always run tests** after making changes: `pytest tests/ -v`
2. **Teff uses log-space normalization** - different from other parameters
3. **BatchNorm requires batch_size > 1** during training
4. **Model outputs are normalized** - use LabelNormalizer.inverse_transform()
5. **Uncertainty predictions are log-scale** - apply exp() via loss.get_predicted_scatter()
6. **Multi-survey**: Each survey has its own encoder; stars can have data from multiple surveys
7. **Multi-label**: Each label source has its own output head; stars may have labels from one or both
8. **has_data masks**: Critical for multi-survey training - indicates which surveys a star has spectra from
9. **gaia_id alignment**: All data aligned by Gaia DR3 source ID for cross-matching
10. **TrainingHistory**: Contains per-survey and per-labelset metrics for visualization
11. **Saliency analysis**: Two complementary approaches in `dorothy.analysis`:
    - `SaliencyAnalyzer`: Gradient-based (Jacobian) saliency maps
    - `AblationSaliencyAnalyzer`: Sliding window ablation with training-distribution weighting
12. **DynamicInputMasking**: Uses random offset to shift block boundaries, preventing fixed positional patterns
13. **Evaluate command**: Uses `evaluate_on_test_set()` from `evaluation_utils.py` to ensure metrics match training validation exactly. Saves data split indices (`data_split.pkl`) during training for reproducible evaluation.
14. **Normalized vs Physical space**: Evaluation reports metrics in both spaces - normalized for comparison with training validation, physical for interpretability
15. **Line Spread Functions**: Survey-specific LSF/resolution documented in `docs/line_spread_functions.md`. DESI provides resolution matrices; BOSS provides Gaussian σ (wdisp); LAMOST uses approximate Gaussian with R≈1800 (LRS) or R=7500 (MRS)
16. **Catalogue versions**: v1 catalogues (APOGEE-only) are archived in `data/archive/`. Current v2 supports both APOGEE and GALAH labels with unified label groups.
17. **Raw data sources**: Training data comes from pre-cross-matched FITS files in `data/raw_catalogues/`, not the legacy `.npy` files in `data/raw/`.

## Super-Catalogue Schema (v2)

```
super_catalogue.h5
├── metadata/
│   ├── gaia_id (N,) int64 - Primary key
│   ├── ra (N,) float64
│   └── dec (N,) float64
├── surveys/
│   ├── desi/ (flux, ivar, wavelength, snr)
│   ├── boss/ (flux, ivar, wavelength, snr)
│   ├── lamost_lrs/ (flux, ivar, wavelength, snr)
│   └── lamost_mrs/ (flux_b/r, ivar_b/r, wavelength_b/r, snr)
└── labels/
    ├── apogee/ (values, errors, flags, gaia_id)
    └── galah/ (values, errors, flags, gaia_id)
```

See `docs/super_catalogue.md` for complete documentation.

## Catalogue Building

### Current Pipeline (v2)

```bash
# Build from pre-cross-matched FITS files in data/raw_catalogues/
python scripts/build_catalogue_v2.py --output data/super_catalogue.h5

# Dry run (print statistics only)
python scripts/build_catalogue_v2.py --dry-run
```

| Script | Purpose |
|--------|---------|
| `build_catalogue_v2.py` | Build unified multi-survey, multi-label catalogue |
| `create_deduplicated_catalogue.py` | Remove duplicate entries from catalogue |

### Legacy Pipeline (v1) - Deprecated

Scripts in `scripts/legacy/` built APOGEE-only catalogues from `.npy` files. See `scripts/legacy/README.md` for details.

## CI/CD and Code Quality

- **Pre-commit hooks**: Black formatting and Ruff linting
- **GitHub Actions**: CI runs on push/PR to main
- **Python versions**: 3.10, 3.11, 3.12

Setup:
```bash
pip install pre-commit
pre-commit install
```

## Development Workflow

1. Create/modify module in `dorothy/`
2. Write tests in `tests/test_<module>.py`
3. Run tests: `pytest tests/ -v`
4. Check coverage: `pytest tests/ --cov=dorothy`
5. Format: `black dorothy/ tests/`
6. Lint: `ruff check dorothy/ tests/`
