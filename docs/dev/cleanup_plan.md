# DOROTHY cleanup plan (pre-experiment)

Goal: get `main` into a correct, consistent, well-tested state, with Diego's
improvements from `dorothy_update` ported in properly, **before** any new
modelling experiments. Diego's branch is left untouched; his changes are
re-implemented on `main` (credited via `Co-Authored-By`).

All findings below were verified by reading code; most also by a repro or
failing test. Severity: **[B]** bug, **[R]** risk, **[Q]** quality-of-life.

Baseline: `main` = 652 pass / 1 fail. `dorothy_update` = 51 fail / 30 error.

---

## Phase 0 — Tooling & baseline

- [Q] Add `black` to `[dev]` extras; add `[tool.black] line-length = 100` to match ruff.
- [Q] CI lint job: make it check-only (`black --check`, `ruff check`) instead of
  auto-committing to `main`/PR heads (breaks on forks, never re-tested, unpinned
  versions disagree with pre-commit). Pin versions to match pre-commit.
- [B] `numpy>=1.24` floor is wrong: `np.bitwise_count` needs NumPy 2.0 → bump floor
  (or replace with a portable popcount).
- [B] Fix failing test: `load_merged_sparse` never calls `_check_for_duplicates`.

## Phase 1 — Port Diego's work (done properly)

- **Data-driven parameter list** (replaces hard-coded 11→20):
  - Loader reads `parameter_names` from catalogue attrs (fallback: label width);
    remove `n_params = 11` (catalogue_loader ×4), `range(11)`, literal defaults.
  - CLI auto-populates parameter names / `n_parameters` into the config (like
    `survey_wavelengths`), saved in the checkpoint `config.yaml`.
  - One canonical name list; display names derived from names
    (`mg_h`→`[Mg/H]`, `mg_fe`→`[Mg/Fe]`); saliency/plots/evaluator take names
    from the model/checkpoint, not a module constant.
  - Both old 11-param `[X/Fe]` and new 20-param `[X/H]` catalogues work.
- **Per-star spectral normalization** (`normalize_spectra`, vectorised, ignores bad
  pixels) applied identically in *every* path: sparse, dense, single-survey,
  FITS/predict, saliency examples. Controlled by a config flag (default on),
  recorded in the checkpoint so inference matches training. Also fixes [R] NaN
  flux poisoning a whole spectrum in the old `normalize_spectrum`.
- `head_dropout` per-labelset dropout (fix `deafult_factory` typo → optional field);
  thread through every model constructor (trainer, CLI evaluate, Predictor).
- `LabelNormalizer.fit`: all-masked parameter → identity stats + warning (needed
  when a parameter exists for GALAH but not APOGEE, e.g. Ba/Sr).
- Input masking that drops a survey must also clear `has_data` — implemented in
  `DynamicInputMasking` itself so all paths get it (Diego fixed sparse only).
- Catalogue builder: merge `build_catalogue_ALL_GALAH.py` into
  `build_catalogue_v2.py` as options (`--x-h`, extended element set) instead of a
  forked copy; propagate [X/H] errors.
- Evaluation plots (pred-vs-truth, residuals, histograms) in a new
  `visualization/evaluation_plots.py`: grid sized to n_params, percentile ranges
  (overridable), label-source-aware axis labels, no scipy; plus per-parameter
  z-score histograms. Wired into `dorothy evaluate` for all modes and all heads.
- Drop: hard-coded `survey='desi'` in kNN, `label_sources[1]` as primary, stray
  prints, dead `overall_loss`, half-finished evaluate block.
- Update examples (variant5 etc.) for new options.

## Phase 2 — Trainer consolidation (the revamp; public API unchanged)

Write characterization tests first (sparse path currently has **zero** tests),
then collapse the four near-duplicate fit loops into: data module per input kind
(dense / sparse / single) → one `Batch` type → one `train_step`/`eval_step` →
one epoch loop (scheduler, early stopping, best weights, AMP, checkpoints,
history). `fit`, `fit_multi_survey`, `fit_multi_survey_sparse`,
`fit_multi_labelset`, `TrainingHistory` fields and checkpoint files keep their
signatures/keys. Bugs fixed structurally:

- [B] ReduceLROnPlateau crashes (`step()` without metric) in all multi modes.
- [B] Sparse multi-label: heads averaged equally → each GALAH label weighted
  ~N_APOGEE/N_GALAH× more than an APOGEE label (likely contributor to early
  GALAH overfitting). Unify with dense path (per-label weighting) + optional
  per-source weights in config.
- [B] Sparse multi-label: validation loss / best checkpoint uses primary head
  (APOGEE) only. Use all heads; record per-head best epochs.
- [B] `LossType.MSE` unusable (shape mismatch).
- [B] `fit()`/`fit_multi_survey()` don't impute masked labels before
  `transform` → `log10(0)` → NaN loss with `max_flag_bits: -1`. One shared label
  normalization routine for all modes (also used by evaluation).
- [R] Predicted-scatter clamp only in single mode → one helper everywhere.
- [Q] Redundant forward passes (2× full validation in sparse multi-label;
  n_sources× trunk forwards per batch in dense multi-label); AMP only on some
  paths; LR history per-step vs per-epoch; 0- vs 1-based epoch logs.
- Save `best_epoch`/`best_val_loss` in history pickle.

## Phase 3 — Inference, evaluation, CLI

- [B] `evaluate_on_dense_data` passes `has_data` positionally → treated as
  `survey`; every star "has" every survey → wrong metrics.
- [B] `dorothy predict` feeds 2-channel `[flux, ivar]`; models are trained on
  3-channel `[flux, sigma, mask]` → shape error on every trained model.
- [B] `Predictor` MLP-only; infers architecture with lexicographic key sort
  (wrong for >3 hidden layers — i.e. the default); ignores `config.yaml`
  (activation/norm/scatter_floor) → silent wrong predictions. Rebuild from
  `config.yaml`; support MultiHeadMLP (survey + label source selection).
- [B] Single-survey split filter: train uses `mask.any`, evaluate /
  `_recreate_data_split` use `mask.all` → test indices point at wrong stars.
  One shared split helper used by train and evaluate.
- [B] Z-scores add the scatter floor twice (and in physical units) → one
  consistent definition shared by trainer and evaluator.
- [B] `Evaluator` exclusive-mode survey IDs mislabelled when non-contiguous.
- [B] `SurveyEvaluationResult.summary` KeyError `log_teff` for empty surveys.
- [Q] `predict --batch-size/--denormalize` ignored; `--resume` declared but
  unused (implement or remove); `info` mislabels MultiHeadMLP dims; device
  resolution copy-pasted ×5; dense/single evaluation not batched (GPU OOM risk);
  evaluation scores primary head only → all heads.
- kNN anomaly: restore public `get_embeddings` path, make it multi-head aware.

## Phase 4 — Data loading, analysis, scripts

- [B] Real v2 catalogues with LAMOST MRS: `get_info()`/`load()`/
  `get_survey_wavelengths()` read missing `wavelength` key → `info()` and
  `load(survey=None)` crash for the whole catalogue.
- [B] Dense MRS path keeps padded arms (2·max(n_b,n_r)) while sparse trims
  (n_b+n_r) and encoders are sized n_b+n_r → shape mismatch.
- [B] `load_multi(mode="intersection")` is a no-op on v2.
- [B] Sparse primary-source `has_labels` ignores flags.
- [B] `create_deduplicated_catalogue.py` only knows v1 label names → empty
  output + ZeroDivisionError on v2 (and the loader tells users to run it).
- [B] Ablation saliency "training distribution" weighting double-counts 1/b →
  effectively single-pixel ablation; docstring/default mismatches.
- [R] Builder: a star whose first matched label row is all-NaN locks in an empty
  row; MRS SNR from blue arm only drops red-only stars.
- [R] BatchNorm stats include zero rows for missing surveys (LayerNorm default OK).
- [R] `np.arange` fallback gaia_ids collide across surveys.
- [Q] Consolidate duplicated label-source resolution / remap / ivar→sigma code;
  plots use last epoch instead of best epoch; `--verbose` can't be disabled.

## Phase 5 — Docs & assistant rules

- Rewrite `CLAUDE.md`: accurate config examples (current examples use
  non-existent `survey`/`label_source`/`architecture` keys that are silently
  ignored), file tree, conventions; drop hard-coded test counts/coverage tables
  that go stale; add "invariants" section (train/inference preprocessing parity,
  parameter names from catalogue, has_data semantics, checkpoint contract).
- Fix README programmatic example (4 non-existent methods/attrs).
- Make config schema reject unknown keys (`extra="forbid"`) so silent fallbacks
  like the above become errors.
- Update `docs/super_catalogue.md` for v2/[X/H]/20-param catalogues.

---

## After cleanup (science work, separate PRs, each behind config flags)

1. Post-hoc per-head/per-parameter uncertainty recalibration on validation.
2. EMA of weights; β-NLL option; weight decay excluding norm/bias params.
3. Input error channel: bounded precision instead of raw σ (masked σ=0 currently
   looks like infinite precision); optional continuum normalization.
4. Speed: `torch.compile`, fewer validation passes.
5. Diagnostic: APOGEE–GALAH overlap error calibration (Var(Δ) vs σ_A²+σ_G²).
