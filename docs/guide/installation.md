# Installation

DOROTHY needs Python 3.10 or later, PyTorch 2.0 or later and NumPy 2.0 or
later.

```bash
git clone https://github.com/joshspeagle/dorothy.git
cd dorothy
pip install -e ".[dev]"
dorothy --version
```

The `dev` extra adds the test and lint tools. To train on a GPU, first install
the PyTorch build for your CUDA version (see
[pytorch.org](https://pytorch.org/get-started/locally/)). Apple silicon GPUs
work through `device: mps`.

## Building this documentation

```bash
pip install --group docs                  # pip 25.1 or later
sphinx-build -W --keep-going docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser. `-W` turns warnings,
including malformed docstrings, into errors, as the CI build does.
