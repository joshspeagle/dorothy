# Installation

DOROTHY needs Python 3.10 or later, PyTorch 2.0 or later and NumPy 2.0 or
later.

```bash
git clone https://github.com/joshspeagle/dorothy.git
cd dorothy
pip install -e ".[dev]"
dorothy --version
```

The `dev` extra adds the test and lint tools. For training on a GPU, first
install the PyTorch build for your CUDA version (see
[pytorch.org](https://pytorch.org/get-started/locally/)). Apple silicon GPUs
work through `device: mps`.

## Documentation

To build this site locally (pip 25.1 or later):

```bash
pip install --group docs
mkdocs serve        # serves the site at http://127.0.0.1:8000
```
