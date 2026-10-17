# Changelog

## Unreleased

### Added

- Per-spectrum flux normalization (`data.normalize_spectra`, default true):
  each spectrum is scaled by the median and interquartile range of its valid
  pixels, in every loading path and at prediction time.
- Parameter sets defined by the catalogue: the builder writes 11 [X/Fe]
  parameters by default, [X/H] abundances with `--abundance-scale x_h` and 17
  elements (20 parameters) with `--elements extended`. The parameter names
  set the model's output size.
- `multi_head_model.head_dropout`: dropout per output head.
- `training.label_source_weighting` (`per_label` or `per_source`) and
  `training.label_source_weights` set how the losses of several label
  sources combine.
- `loss: mse` trains with a masked mean squared error.
- `dorothy evaluate` scores every output head and writes evaluation plots
  per label source. `dorothy predict` supports multi-head models
  (`--survey`, `--label-source`).
- `dorothy.data.training_data`: one data path from a configuration to
  training data, splits and batches, shared by `dorothy train`,
  `dorothy evaluate` and the Python API.
- `dorothy.inference.load_checkpoint` and `evaluate_model`.
- `CatalogueLoader.load_spectrum` and `load_labels_for_index` load one star.
- A documentation site built with MkDocs.

### Fixed

- `ReduceLROnPlateau` crashed in every multi-survey training mode.
- Masked labels could produce NaN losses (for example log10(0) for a masked
  Teff with `max_flag_bits: -1`).
- Sparse multi-label validation selected the best epoch from the APOGEE head
  alone.
- z-scores in the evaluator added the scatter floor twice.
- Per-survey metrics mapped non-contiguous integer survey IDs to the wrong
  survey.
- Dense loading of LAMOST MRS used the wrong spectrum width.
- `dorothy predict` required label columns in the input FITS file and fed the
  model a 2-channel input.
- The catalogue builder flagged elements a label source does not measure, so
  `max_flag_bits: 0` masked every star of that source in extended builds.
- The catalogue builder kept a star's first match even when it had no valid
  labels or no valid pixels.
- `load_merged_sparse` did not detect duplicate stars.

### Changed

- The trainer's four training modes share one training loop.
- Unknown configuration keys are errors. `survey` and `label_source` are
  accepted as aliases of `surveys` and `label_sources`.
- `data_split.pkl` stores catalogue rows. Split files from older versions
  are converted when loaded.
- The deduplication script moved to `scripts/legacy/`; v2 catalogues have
  one row per star.

### Behaviour changes

- Stars with at least one valid label are used in sparse single-label
  training; before, every label had to be valid. Existing configurations
  select more stars and get a different split.
- With several label sources, the label normalization is fit on all of them
  (the first source wins where several label a star), and the default loss
  weighting is `per_label`. Use `per_source` for the previous equal weighting
  of heads.
- The validation loss of a multi-label model combines all heads, so the best
  epoch can change.
- `final_model.pth` holds the weights of the last epoch; before, it held the
  best weights, like `best_model.pth`.
- `dorothy predict` writes physical units by default (`--normalized` for
  model units).
- The catalogue builder keeps each star's highest-S/N spectrum and its label
  row with the most valid labels, computes the LAMOST MRS S/N over both arms
  and zeroes non-finite pixels. Rebuilt catalogues differ for stars with
  repeat observations.
- Learning rates are logged per optimizer step in every training mode.
