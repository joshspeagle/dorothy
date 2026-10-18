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
- `training.amp_dtype` (`bfloat16` by default, or `float16`) selects the
  precision of `training.amp`.
- `loss: mse` trains with a masked mean squared error.
- Label-aware uncertainty: each head predicts the mean with a label decoder
  and the log-scatter with a parallel error decoder, which also reads the
  reported label errors, or a learned missing vector where an error is
  missing. No layer that feeds the mean reads the label errors. Both
  decoders have the same hidden layers: the `output_hidden` layers in the
  multi-head model, and the last hidden layer in the MLP, which the decoders
  take over. `training.label_error_dropout` (default 0.5) sets the fraction
  of training stars that get the missing vector.
  `HeadEvaluation.z_label_errors` holds the label errors that the z-scores
  add to the predicted scatter, which only old checkpoints have.
- `dorothy evaluate` scores every output head and writes evaluation plots
  per label source. `dorothy predict` supports multi-head models
  (`--survey`, `--label-source`).
- `dorothy.data.training_data`: one data path from a configuration to
  training data, splits and batches, shared by `dorothy train`,
  `dorothy evaluate` and the Python API.
- `dorothy.inference.load_checkpoint` and `evaluate_model`.
- `dorothy.analysis.compare_label_sources` compares the APOGEE and GALAH labels
  of the stars that both measure with the spread that their reported errors
  predict, for each parameter. It needs no trained model.
  `plot_label_differences` plots the differences against Teff and [Fe/H].
- `CatalogueLoader.load_spectrum` and `load_labels_for_index` load one star.
  `CatalogueLoader.load_labels` loads one label source for every star,
  matched by Gaia ID.
- A documentation site (Sphinx), with background, methods, usage guides and an API reference.
- The Dorothy logo in the README and on the documentation site.

### Fixed

- Training on Apple MPS failed at the first step with PyTorch before 2.5,
  which rejects autocast on MPS even when it is disabled.
- The catalogue loaders raised h5py's `KeyError`, not `ValueError`, for a
  catalogue without a `labels` group.
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
- `loss: penalty` passed configuration checks but failed when training started; the configuration now rejects it.
- Ablation saliency with `use_training_distribution=True` weighted its
  log-spaced block sizes by 1/b, so the map was close to single-pixel
  ablation: for BOSS, block sizes of at most 5 pixels had 84% of the weight.
  A size that several grid points round to counted only once.
- The saliency analyzers converted the predicted log-scatter ln s to a
  scatter without the clamp of `losses.predicted_scatter`, so their
  uncertainties disagreed with `Predictor` when ln s was outside [-10, 5].
- A NaN or infinity in a masked pixel made the whole saliency map NaN: the
  analyzers multiplied the flux and error by the mask.
- The saliency plots raise a `ValueError` for a fully masked spectrum. Three
  of them failed with an unrelated numpy error, and
  `plot_ablation_saliency_heatmap` drew an empty figure.
- The bar charts of validation metrics in the training plots showed the last
  epoch instead of the best epoch, whose weights `best_model.pth` holds.
- LAMOST MRS has no `wavelength` dataset, so `CatalogueLoader.info`,
  `get_info` and `load(survey=None)` raised `KeyError` for any catalogue with
  MRS, and `load`, `load_raw_spectra` and `get_survey_wavelengths` raised it
  for MRS. `load` now trims each MRS arm to its own grid, like the other
  loaders.
- `CatalogueLoader.load_multi(mode="intersection")` returned every star of a
  v2 catalogue, because each survey has a row for every star.
- `load_merged_sparse` and `load_merged_for_training` counted a star as
  labelled by a source (`has_labels_dict`) even when `max_flag_bits` removed
  all its labels.
- The duplicate-star error told users to run a deduplication script that
  moved to `scripts/legacy/` and handles only v1 catalogues. It now tells
  them to rebuild the catalogue with `scripts/build_catalogue_v2.py`.
- `load_merged` numbered the rows of a catalogue without Gaia IDs from 0, so
  it dropped row 0 and merged unrelated stars that share a row number in
  different surveys. It now raises `ValueError`.
- `load_merged` loaded spectra with a NaN S/N, and raised `KeyError` for such
  a star when no other requested survey had it.
- `load_merged_sparse` read the primary label source in row order without
  checking its Gaia IDs, so labels stored in another order went to the wrong
  stars. It now raises `ValueError` when they differ from `metadata/gaia_id`.
- `load_merged_sparse` and `load_merged_for_training` gave a second label
  source without a `gaia_id` dataset a copy of the first source's labels, so
  its head trained on them. They now raise `ValueError`, naming the label
  group.
- Input and label masking, the `mean` combination of `MultiHeadMLP` and FITS
  prediction with `normalize_spectra: false` multiplied by a mask, so a NaN
  or infinity in a masked entry stayed NaN. They now select with the mask.
  Results for finite inputs are unchanged, except for the sign of zero.
- Gradient saliency used the derivative of the log-scatter ln s for its
  scatter terms, which ignores the scatter floor and the clamp of ln s. It
  now uses the exact derivative of the predicted scatter σ:
  `SaliencyResult.jacobian_sigma_flux` and `jacobian_sigma_error` hold dσ/dx
  and replace `jacobian_lns_flux` and `jacobian_lns_error`.
- If every validation loss was NaN, `best_epoch` was 0, but `best_model.pth`
  and the trained model held the weights from before training. Training now
  raises `RuntimeError` when no epoch has a finite validation loss.
- The epoch checkpoints of `training.save_every` include `config.yaml` and
  `normalizer.pkl`, so `load_checkpoint` rebuilds their model, multi-head
  models included.
- When `MultiHeadMLP` encoded every row of a batch (an encoder with batch
  normalization or dropout in training mode, or
  `encode_present_rows_only = False`), a NaN in a row without that survey's
  data made the weight gradients NaN, and with batch normalization every
  output. These rows now enter the encoder as zeros, as in the trainer's
  batches.

### Performance

- Training steps run about 1.75x faster and validation passes 2 to 2.6x
  faster on CPU: validation computes each head's metrics once, input masking
  runs in one pass, sparse batches gather only the stars that have each
  survey, and multi-head encoders run only on those stars. On CUDA, batches
  are copied asynchronously and a training step synchronizes with the GPU 4
  times instead of about 34.
- Gradient saliency runs about 6 times faster: one batched backward pass
  replaces one pass per output. Ablation saliency runs 6 to 11 times faster:
  when a spectrum reaches the output only through the model's first linear
  layer, ablated spectra reuse that layer's per-pixel sums. k-NN anomaly
  detection runs 15 to 40 times faster with a brute-force search. Saliency
  maps and k-NN distances agree with the previous results to floating-point
  rounding.
- `load_merged_sparse` reads only the rows that have spectra: 5 times faster
  with 43% less peak memory for 40,000 stars in four surveys. Validation
  metrics are computed 3 times faster, and `FITSLoader.load` normalizes all
  spectra in one call (1.5 to 2 times faster). Results are bit-identical,
  except for the sign of zero and of NaN in some evaluator metrics.

### Changed

- The trainer's four training modes share one training loop.
- Unknown configuration keys are errors. `survey` and `label_source` are
  accepted as aliases of `surveys` and `label_sources`.
- `data_split.pkl` stores catalogue rows. Split files from older versions
  are converted when loaded.
- The deduplication script moved to `scripts/legacy/`; v2 catalogues have
  one row per star.
- `AnomalyDetector.tree` is removed; the detector searches the training
  embeddings directly.
- `SparseMergedData.ivar` is replaced by `SparseMergedData.sigma`, which
  holds 1/√ivar and 0 at invalid pixels, so the pixel mask is σ > 0.
  `SparseMergedData.memory_usage_mb` reports `<survey>_sigma` instead of
  `<survey>_ivar`.
- The minimum PyTorch version is 2.4.

### Behaviour changes

- The default embedding of a newly trained MLP
  (`MLP.get_embeddings`, and so `AnomalyDetector`) is the input of its two
  decoders, which take over the last hidden layer: the layer before it, for
  example 256 values instead of 64 for `examples/variant1_boss_apogee.yaml`.
  Old checkpoints keep the last hidden layer. Embeddings are always flat, so
  `layer_index=-1` gives (batch, 2 n_parameters).
- `load_merged_sparse` computes σ = 1/√ivar once, with the numpy code of the
  other loading paths, and sparse batches copy it. Before, each batch
  computed σ with `torch.sqrt`, which is not exactly rounded on the CPU, so σ
  could differ from the other paths. How often depends on the CPU.
- `normalize_spectra` marks a pixel invalid (flux and ivar 0) when the
  scaled ivar or the normalized flux overflows to infinity. Before, such a
  pixel kept an infinite ivar, which gave σ = 0 with mask 1.
- Multi-head encoders run only on the stars that have their survey (except
  encoders with batch normalization or dropout in training mode). Results
  change by floating-point rounding only: losses after a few epochs differ
  by up to 4e-7 relative, less than between runs on 1 and 4 threads.
  `MultiHeadMLP.encode_present_rows_only = False` restores the previous
  computation. A NaN in the input of a survey a star lacks no longer reaches
  the outputs.
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
- Ablation saliency with `use_training_distribution=True` gives every point
  of its block-size grid equal weight (for BOSS, block sizes of at most 5
  pixels now have 24% of the weight), and `AblationSaliencyResult.block_size`
  lists one size per grid point, with repeats.
- The saliency analyzers compute the predicted scatter with
  `losses.predicted_scatter`. Their uncertainties, scatter changes and
  Fisher weights change where ln s is outside [-10, 5].
- Gradient saliency differentiates σ = sqrt(exp(2 ln s) + s₀²):
  dσ/dx = (exp(2 ln s) / σ) d ln s/dx, and 0 where ln s is outside
  [-10, 5]. Its Fisher importance and the scatter panels of its plots change
  most where σ is close to the scatter floor s₀ or ln s is clamped.
- `val_metrics.png`, `zscore_calibration.png`, `per_survey_metrics.png` and
  `per_labelset_metrics.png` in a checkpoint's `plots/` show the best epoch
  instead of the last.
- `CatalogueLoader.load_multi(mode="intersection")` keeps a star only if
  every survey has data for it (`snr > 0`, the rule of the sparse loader).
- `SparseMergedData.has_labels_dict` marks a star only if it has a label
  that survives the flag filter, as its label mask does.
- Dense loading skips spectra with a NaN S/N, as sparse loading does.
- `training.optimizer.weight_decay` applies only to weight matrices; biases
  and normalization weights are no longer decayed.
- The loss variance is s² instead of σ_y² + s², and new models have
  label-aware uncertainty. The predicted scatter s is the expected scatter
  between the prediction and the label source's measurement, label noise
  included; before, it was the scatter beyond the reported label error.
  Validation, evaluation and prediction use the missing vector.
- The z-scores of label-aware models are (μ − y) / s in training validation,
  `evaluate_model` and the `dorothy evaluate` plots. Checkpoints saved before
  label-aware uncertainty keep (μ − y) / sqrt(σ_y² + s²).
- Checkpoint configurations store `label_aware_uncertainty` in the model
  section. A checkpoint without it loads with the additive output layer and
  predicts and evaluates exactly as before. Training rejects
  `label_aware_uncertainty: false`, so `--resume` from such a checkpoint
  fails. `load_checkpoint` without a `config.yaml` builds only the additive
  layer and raises `ValueError` for label-aware weights.
- With `data.normalize_spectra: false`, every loading path and prediction set
  the flux and ivar of invalid pixels to 0, as normalization does. A pixel is
  invalid when its flux is not finite, or its ivar is not finite and
  positive. Before, training passed the stored flux of masked pixels to the
  model, and a NaN flux with a positive ivar reached the model.
  `dorothy.data.zero_invalid_pixels` applies the rule.
- The Fisher-weighted change of ablation saliency is
  ((Δμ)² + 2 (Δσ)²) / σ₀², the quadratic form of gradient saliency for
  finite changes, with σ from `losses.predicted_scatter`. Its scatter term
  was 2 (Δ ln s)², which ignores the scatter floor and the clamp of ln s.
  `AblationSaliencyResult.fisher_weighted` changes most where σ is close to
  the floor or ln s is clamped.
- With `training.amp: true`, the training forward pass on CUDA runs in
  bfloat16 without a gradient scaler, instead of float16 with one;
  `training.amp_dtype: float16` keeps float16. On a GPU without native
  bfloat16 (compute capability below 8.0, such as a V100 or T4), such a
  configuration now raises `ValueError` and needs `training.amp_dtype:
  float16`. Validation runs in float32 instead of under autocast, which can
  change the best epoch.
