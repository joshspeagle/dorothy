# DOROTHY

Deep learning framework for inferring stellar parameters from spectroscopic data.

## Overview

DOROTHY uses neural networks to predict stellar atmospheric parameters and chemical abundances from observed spectra. It supports multiple astronomical surveys including DESI, BOSS, LAMOST, and APOGEE.

### Predicted Parameters

- **Atmospheric**: Teff, log g, [Fe/H]
- **Abundances**: [Mg/Fe], [C/Fe], [Si/Fe], [Ni/Fe], [Al/Fe], [Ca/Fe], [N/Fe], [Mn/Fe]

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

```python
import torch
from dorothy.models import MLP
from dorothy.losses import HeteroscedasticLoss
from dorothy.data import LabelNormalizer

# Create model
model = MLP(
    input_features=15300,  # 2 channels × 7650 wavelengths
    output_features=22,    # 11 params + 11 uncertainties
    hidden_layers=[5000, 2000, 1000, 500, 200, 100],
)

# Create loss function
loss_fn = HeteroscedasticLoss(scatter_floor=0.01, n_parameters=11)

# Forward pass
x = torch.randn(32, 2, 7650)  # Batch of spectra
output = model(x)             # Shape: (32, 22)

# Compute loss (target includes labels + errors)
target = torch.randn(32, 22)
target[:, 11:] = torch.abs(target[:, 11:]) + 0.01  # Positive errors
loss = loss_fn(output, target)
```

## Configuration

DOROTHY uses Pydantic for configuration validation:

```python
from pathlib import Path
from dorothy.config import ExperimentConfig, DataConfig

config = ExperimentConfig(
    name="my_experiment",
    data=DataConfig(
        fits_path=Path("/data/spectra.fits"),
        survey="desi",
    ),
    seed=42,
)
```

## Architecture

The default MLP architecture:

```
Input:  (batch, 2, 7650) → Flatten → (batch, 15300)
Hidden: 15300 → 5000 → 2000 → 1000 → 500 → 200 → 100
Output: (batch, 22) → 11 predictions + 11 uncertainties
```

Each hidden layer uses:
- Linear transformation
- Batch normalization (or Layer normalization)
- GELU activation

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
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

## License

MIT License - see LICENSE file for details.
