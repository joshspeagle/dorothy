# Install

DOROTHY needs Python 3.10 or later, PyTorch 2.4 or later and NumPy 2.0 or
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
