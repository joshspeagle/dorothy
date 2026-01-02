# DOROTHY

Deep learning framework for inferring stellar parameters from spectroscopic data.

## Overview

DOROTHY uses neural networks to predict stellar atmospheric parameters and chemical abundances from observed spectra. It supports multiple astronomical surveys and training configurations:

- **Single-survey training**: Train on one spectroscopic survey (BOSS, DESI, LAMOST)
- **Multi-survey training**: Combine spectra from multiple surveys with shared representations
- **Multi-label training**: Use labels from multiple sources (APOGEE, GALAH) with separate output heads

### Predicted Parameters

| Parameter | Description | Units |
|-----------|-------------|-------|
| Teff | Effective temperature | K |
| log g | Surface gravity | log(cm/s²) |
| [Fe/H] | Iron abundance | dex |
| [Mg/Fe] | Magnesium | dex |
| [C/Fe] | Carbon | dex |
| [Si/Fe] | Silicon | dex |
| [Ni/Fe] | Nickel | dex |
| [Al/Fe] | Aluminum | dex |
| [Ca/Fe] | Calcium | dex |
| [N/Fe] | Nitrogen | dex |
| [Mn/Fe] | Manganese | dex |

### Supported Surveys

| Survey | Wavelength Bins | Resolution |
|--------|-----------------|------------|
| DESI | 7,650 | R~3000-5000 |
| BOSS | 4,506 | R~2000 |
| LAMOST LRS | 3,473 | R~1800 |
| LAMOST MRS | 3,375 × 2 arms | R~7500 |

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/dorothy.git
cd dorothy

# Install in development mode
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.0
- See `pyproject.toml` for full dependencies

## Quick Start

### Training with Configuration File

```bash
# Single-survey training
dorothy train examples/variant1_boss_apogee.yaml

# Multi-survey training
dorothy train examples/variant2_multi_survey.yaml

# Multi-label training
dorothy train examples/variant3_multi_labelset.yaml
```

### Programmatic Usage

```python
from dorothy.data import CatalogueLoader
from dorothy.models import MultiHeadMLP
from dorothy.training import Trainer
from dorothy.config import ExperimentConfig

# Load data from super-catalogue
loader = CatalogueLoader("data/super_catalogue.h5")
loader.info()  # Print summary

# Load multi-survey data
merged = loader.load_merged(surveys=["boss", "desi"])
X, y, has_data, _ = loader.load_for_multi_survey_training(
    surveys=["boss", "desi"],
    label_source="apogee"
)

# Train using config file
config = ExperimentConfig.from_yaml("examples/variant2_multi_survey.yaml")
trainer = Trainer(config)
history = trainer.fit_multi_survey(X, y, has_data)
```

## Architecture

### Standard MLP

For single-survey training:

```
Input:  (batch, 3, wavelengths) → Flatten → (batch, 3*wavelengths)
Hidden: input_dim → 5000 → 2000 → 1000 → 500 → 200 → 100
Output: (batch, 22) → 11 predictions + 11 uncertainties
```

### Multi-Head MLP

For multi-survey and/or multi-label training:

```
Survey inputs → SurveyEncoders → SharedTrunk → OutputHeads → Predictions
```

- **SurveyEncoder**: One per survey, handles different wavelength grids
- **SharedTrunk**: Combines representations (concat or mean)
- **OutputHead**: One per label source (APOGEE, GALAH)

## Configuration

DOROTHY uses YAML configuration files. Example for multi-survey training:

```yaml
name: multi_survey_experiment

data:
  catalogue_path: data/super_catalogue.h5
  surveys: [boss, desi]
  label_source: apogee
  max_flag_bits: 0

model:
  architecture: multi_head
  latent_dim: 256
  encoder_hidden: [1024, 512]
  trunk_hidden: [512, 256]
  output_hidden: [64]
  combination_mode: concat
  normalization: layernorm
  activation: gelu

training:
  epochs: 100
  batch_size: 512
  learning_rate: 0.001
  loss: heteroscedastic
  gradient_clip: 10.0
  optimizer:
    type: adamw
    weight_decay: 0.01
  scheduler:
    type: one_cycle
    max_lr: 0.001

output_dir: outputs
seed: 42
```

## Super-Catalogue

The unified HDF5 catalogue provides row-matched data from multiple surveys:

```python
from dorothy.data import CatalogueLoader

loader = CatalogueLoader("data/super_catalogue.h5")
info = loader.get_info()

print(f"Total stars: {info.n_stars}")
print(f"Surveys: {info.survey_names}")
print(f"Label sources: {info.label_sources}")
```

See [docs/super_catalogue.md](docs/super_catalogue.md) for the complete schema and usage guide.

## Saliency Analysis

Visualize which wavelengths are most important for each stellar parameter prediction. DOROTHY provides two complementary approaches:

### Gradient-Based Saliency

Computes sensitivity maps using the Jacobian of model outputs with respect to inputs, weighted by Fisher information:

```python
from dorothy.analysis import SaliencyAnalyzer, plot_parameter_saliency

# Load model and create analyzer
analyzer = SaliencyAnalyzer(model, device="cuda")

# Compute saliency for a spectrum
result = analyzer.compute_saliency(X, wavelength, survey="boss")

# Visualize per-parameter importance
fig = plot_parameter_saliency(result, "teff", output_path="teff_saliency.png")
```

See `examples/saliency_example.py` for complete usage.

### Ablation-Based Saliency

Measures importance by masking contiguous spectral blocks and observing prediction changes. Supports averaging over multiple block sizes weighted by the training distribution (1/b weighting):

```python
from dorothy.analysis import AblationSaliencyAnalyzer, plot_ablation_parameter_saliency

analyzer = AblationSaliencyAnalyzer(model, device="cuda")

# Use training distribution weighting (recommended)
result = analyzer.compute_saliency(
    X, wavelength, survey="boss",
    use_training_distribution=True,
    n_block_sizes=21,
    f_max=0.5
)

fig = plot_ablation_parameter_saliency(result, "teff", output_path="teff_ablation.png")
```

See `examples/ablation_saliency_example.py` for complete usage.

## CLI Commands

```bash
# Train a model
dorothy train config.yaml

# Make predictions
dorothy predict --checkpoint ./model --input data.fits --output predictions.csv

# Show model info
dorothy info ./model

# Evaluate on held-out test set
dorothy evaluate ./model

# Evaluate with options
dorothy evaluate ./model --model final_model.pth  # Use final instead of best model
dorothy evaluate ./model --output results.json --format json  # Save as JSON
dorothy evaluate ./model --output results.md --format markdown  # Save as Markdown
```

### Evaluate Command

The `evaluate` command computes metrics on the held-out test set (10% of data by default). It uses the exact same evaluation pipeline as training validation, ensuring consistent z-score metrics.

**Output includes:**

- **Normalized space metrics**: Comparable to training validation (z_scatter ~1.0 for well-calibrated uncertainties)
- **Physical space metrics**: Interpretable units (e.g., Teff RMSE in K, [Fe/H] RMSE in dex)

**Metrics computed:**

- RMSE, Bias, Standard Deviation
- MAE, Median Offset, Robust Scatter (IQR-based)
- Z-score calibration (z_median, z_scatter)
- Predicted uncertainty percentiles (p16, p50, p84)

## Testing

```bash
# Run all tests (653 tests)
pytest tests/ -v

# Run with coverage (88% coverage)
pytest tests/ --cov=dorothy --cov-report=html
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Format code
black dorothy/ tests/

# Lint code
ruff check dorothy/ tests/

# Type check
mypy dorothy/
```

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

## Project Structure

```
dorothy/
├── dorothy/
│   ├── config/          # Configuration schema
│   ├── data/            # Data loading and preprocessing
│   ├── models/          # Neural network architectures
│   ├── losses/          # Loss functions
│   ├── training/        # Training loop
│   ├── inference/       # Prediction, evaluation, and test set metrics
│   ├── analysis/        # Anomaly detection, saliency analysis
│   ├── visualization/   # Training plots
│   └── cli/             # Command-line interface
├── scripts/             # Catalogue building scripts
├── examples/            # Example configurations
├── tests/               # Test suite
└── docs/                # Documentation
```

## License

MIT License - see LICENSE file for details.
