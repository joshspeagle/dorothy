<p align="center">
  <img src="docs/_static/dorothy.svg" alt="Dorothy the dinosaur: a teal sauropod with gold star spots, standing on a stellar spectrum" width="240">
</p>

# DOROTHY

DOROTHY infers stellar parameters, chemical abundances and their uncertainties
from spectra with neural networks.

- **Inputs:** spectra from DESI, BOSS, LAMOST LRS and LAMOST MRS, alone or
  combined.
- **Labels:** APOGEE and GALAH parameters, with one output head per label
  source.
- **Uncertainties:** each prediction comes with a predicted scatter, the
  expected difference from the label source's measurement, learned with the
  reported label errors as an input and calibrated on the validation stars.

**Documentation: <https://joshspeagle.com/dorothy/>**

## Installation

```bash
git clone https://github.com/joshspeagle/dorothy.git
cd dorothy
pip install -e ".[dev]"
```

DOROTHY needs Python 3.10 or later, PyTorch 2.4 or later and NumPy 2.0 or
later.

## Usage

```bash
python scripts/build_catalogue_v2.py                  # build data/super_catalogue.h5
dorothy train examples/variant2_multi_survey.yaml     # train; writes a checkpoint directory
dorothy evaluate <checkpoint_dir>                     # metrics and plots on the test split
dorothy predict --checkpoint <checkpoint_dir> --input spectra.fits --survey boss
```

The [quick start](https://joshspeagle.com/dorothy/guide/quickstart.html) runs these
steps and shows the Python API. `examples/` has configurations for single- and
multi-survey models with one or two label sources.

## Development

```bash
python -m pytest tests/ -q
black --check dorothy tests scripts examples && ruff check dorothy tests scripts examples
pip install --group docs && sphinx-build -W --keep-going docs docs/_build/html   # build the documentation
```

See [`CLAUDE.md`](CLAUDE.md) for the project's conventions and
[`CHANGELOG.md`](CHANGELOG.md) for changes.

## License

MIT License. See [`LICENSE`](LICENSE).
