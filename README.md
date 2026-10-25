<p align="center">
  <img src="docs/_static/dorothy.svg" alt="Dorothy the dinosaur: a teal sauropod with gold star spots, standing on a stellar spectrum" width="240">
</p>

# DOROTHY

DOROTHY infers stellar parameters, chemical abundances and their uncertainties
from spectra with neural networks. It reads spectra from DESI, BOSS, LAMOST LRS
and LAMOST MRS, alone or combined, and predicts the labels of APOGEE and GALAH.
Each prediction comes with a calibrated uncertainty.

**Documentation: <https://joshspeagle.com/dorothy/>**

## Try it

```bash
git clone https://github.com/joshspeagle/dorothy.git
cd dorothy
pip install -e .
python examples/synthetic_quickstart.py   # writes a small synthetic catalogue
dorothy train quickstart/config.yaml      # trains in seconds on a CPU
```

To train on a GPU, first install the PyTorch build for your CUDA version
([Install](https://joshspeagle.com/dorothy/start/install.html)). The
[quick start](https://joshspeagle.com/dorothy/start/quickstart.html) continues
with evaluation and prediction.

## Read more

| To | Go to |
|----|-------|
| Learn what DOROTHY does | [Start](https://joshspeagle.com/dorothy/start/what-it-does.html) |
| Learn what the catalogue holds | [The catalogue](https://joshspeagle.com/dorothy/catalogue/contents.html) |
| Predict, train or evaluate | [Guides](https://joshspeagle.com/dorothy/guides/predict.html) |
| Understand the methods | [How it works](https://joshspeagle.com/dorothy/methods/overview.html) |
| Look up a command or setting | [Reference](https://joshspeagle.com/dorothy/reference/commands.html) |
| Change the code or the docs | [Contribute](https://joshspeagle.com/dorothy/contribute/code-map.html) |

[`CLAUDE.md`](https://github.com/joshspeagle/dorothy/blob/main/CLAUDE.md)
holds the project's conventions and the test and lint commands that CI runs.
[`CHANGELOG.md`](https://github.com/joshspeagle/dorothy/blob/main/CHANGELOG.md)
lists the changes since the last release.

## License

MIT License. See [`LICENSE`](https://github.com/joshspeagle/dorothy/blob/main/LICENSE).
