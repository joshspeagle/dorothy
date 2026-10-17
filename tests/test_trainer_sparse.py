"""Tests for sparse multi-survey training and behaviour shared by all fit modes."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dorothy.config.schema import (
    DataConfig,
    ExperimentConfig,
    MultiHeadModelConfig,
    SchedulerConfig,
    SchedulerType,
    TrainingConfig,
)
from dorothy.data.catalogue_loader import SparseMergedData
from dorothy.training.trainer import EVALUATOR_METRIC_NAMES, Trainer

N_TOTAL = 160
N_PARAMS = 4
WAVES = {"boss": 40, "desi": 30}
PARAMS = ["teff", "logg", "fe_h", "mg_h"]


def _labels(rng, latent, n_params=N_PARAMS, noise=0.05):
    """Labels that are a learnable function of the latent star properties."""
    values = np.zeros((len(latent), n_params), dtype=np.float32)
    values[:, 0] = 5000 + 800 * latent[:, 0]
    values[:, 1] = 3 + 0.8 * latent[:, 1]
    values[:, 2] = 0.3 * latent[:, 2]
    values[:, 3] = 0.2 * latent[:, 0] - 0.1 * latent[:, 2]
    errors = np.full_like(values, noise)
    errors[:, 0] = 50.0
    values += rng.normal(size=values.shape).astype(np.float32) * errors
    return np.stack([values, errors, np.ones_like(values)], axis=1)


def make_sparse_data(multi_label: bool = False, seed: int = 0) -> SparseMergedData:
    """Synthetic SparseMergedData: partial survey coverage, optional 2 label sources."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(N_TOTAL, 3)).astype(np.float32)

    has = {
        "boss": rng.uniform(size=N_TOTAL) < 0.8,
        "desi": rng.uniform(size=N_TOTAL) < 0.5,
    }
    has["boss"][~(has["boss"] | has["desi"])] = True  # every star has a survey

    flux, ivar, g2l, l2g = {}, {}, {}, {}
    for survey, n_wave in WAVES.items():
        idx = np.where(has[survey])[0]
        basis = rng.normal(size=(3, n_wave)).astype(np.float32)
        f = latent[idx] @ basis + 1.0
        f += rng.normal(size=f.shape).astype(np.float32) * 0.05
        iv = np.full_like(f, 100.0)
        iv[:, :3] = 0.0  # always-masked pixels
        flux[survey], ivar[survey] = f, iv
        g = np.full(N_TOTAL, -1, dtype=np.int32)
        g[idx] = np.arange(len(idx), dtype=np.int32)
        g2l[survey], l2g[survey] = g, idx.astype(np.int32)

    labels = _labels(rng, latent)
    labels_dict = has_labels = sources = None
    if multi_label:
        galah = _labels(rng, latent, noise=0.08)
        galah_has = rng.uniform(size=N_TOTAL) < 0.4
        galah[~galah_has, 2, :] = 0.0
        galah[~galah_has, :2, :] = 0.0
        apogee_has = rng.uniform(size=N_TOTAL) < 0.8
        labels[~apogee_has, 2, :] = 0.0
        labels[~apogee_has, :2, :] = 0.0
        labels_dict = {"apogee": labels.copy(), "galah": galah}
        has_labels = {"apogee": apogee_has, "galah": galah_has}
        sources = ["apogee", "galah"]

    return SparseMergedData(
        flux=flux,
        ivar=ivar,
        wavelengths={s: np.linspace(4000, 9000, n) for s, n in WAVES.items()},
        snr={s: np.full(len(flux[s]), 20.0) for s in WAVES},
        global_to_local=g2l,
        local_to_global=l2g,
        labels=labels,
        gaia_ids=np.arange(1, N_TOTAL + 1),
        ra=np.zeros(N_TOTAL),
        dec=np.zeros(N_TOTAL),
        surveys=list(WAVES),
        n_total=N_TOTAL,
        n_params=N_PARAMS,
        labels_dict=labels_dict,
        has_labels_dict=has_labels,
        label_sources=sources,
        parameter_names=PARAMS,
    )


def make_config(tmp_path, multi_label=False, epochs=4, **training) -> ExperimentConfig:
    sources = ["apogee", "galah"] if multi_label else ["apogee"]
    return ExperimentConfig(
        name="sparse_test",
        output_dir=tmp_path,
        data=DataConfig(
            catalogue_path=tmp_path / "cat.h5",
            surveys=list(WAVES),
            label_sources=sources,
            parameter_names=PARAMS,
        ),
        multi_head_model=MultiHeadModelConfig(
            survey_wavelengths=WAVES,
            latent_dim=16,
            encoder_hidden=[32],
            trunk_hidden=[32],
            output_hidden=[16],
            label_sources=sources if multi_label else None,
        ),
        training=TrainingConfig(epochs=epochs, batch_size=32, **training),
        device="cpu",
    )


def split(n=N_TOTAL, seed=0):
    perm = np.random.default_rng(seed).permutation(n)
    return perm[: int(0.7 * n)], perm[int(0.7 * n) :]


class TestSparseSingleLabel:
    def test_history_populated(self, tmp_path):
        trainer = Trainer(make_config(tmp_path))
        train_idx, val_idx = split()
        history = trainer.fit_multi_survey_sparse(
            make_sparse_data(), train_idx, val_idx
        )

        assert len(history.train_losses) == len(history.val_losses) == 4
        assert all(np.isfinite(history.train_losses))
        assert all(np.isfinite(history.val_losses))
        assert history.best_epoch == int(np.argmin(history.val_losses))
        assert history.best_val_loss == min(history.val_losses)
        assert history.parameter_names == PARAMS
        assert set(history.val_metrics) == set(EVALUATOR_METRIC_NAMES)
        assert len(history.val_metrics["rmse"]) == 4
        assert history.val_metrics["rmse"][0].shape == (N_PARAMS,)
        assert history.survey_names == list(WAVES)
        for survey in WAVES:
            assert len(history.per_survey_val_losses[survey]) == 4
            assert len(history.per_survey_val_metrics[survey]["rmse"]) == 4
        assert len(history.learning_rates) > 0
        assert len(history.grad_norms) == 4

    def test_loss_decreases(self, tmp_path):
        trainer = Trainer(make_config(tmp_path, epochs=15, learning_rate=3e-3))
        train_idx, val_idx = split()
        history = trainer.fit_multi_survey_sparse(
            make_sparse_data(), train_idx, val_idx
        )
        assert history.train_losses[-1] < history.train_losses[0]
        assert history.best_val_loss < history.val_losses[0]

    def test_best_weights_restored(self, tmp_path):
        trainer = Trainer(make_config(tmp_path, epochs=6))
        train_idx, val_idx = split()
        history = trainer.fit_multi_survey_sparse(
            make_sparse_data(), train_idx, val_idx
        )
        best = trainer._best_weights
        for name, tensor in trainer.model.state_dict().items():
            assert torch.equal(tensor, best[name]), name
        assert history.best_epoch < 6

    def test_reproducible(self, tmp_path):
        train_idx, val_idx = split()
        h1 = Trainer(make_config(tmp_path)).fit_multi_survey_sparse(
            make_sparse_data(), train_idx, val_idx
        )
        h2 = Trainer(make_config(tmp_path)).fit_multi_survey_sparse(
            make_sparse_data(), train_idx, val_idx
        )
        np.testing.assert_allclose(h1.train_losses, h2.train_losses)
        np.testing.assert_allclose(h1.val_losses, h2.val_losses)

    def test_input_data_not_modified(self, tmp_path):
        data = make_sparse_data()
        labels_before = data.labels.copy()
        flux_before = {s: f.copy() for s, f in data.flux.items()}
        train_idx, val_idx = split()
        Trainer(make_config(tmp_path, epochs=1)).fit_multi_survey_sparse(
            data, train_idx, val_idx
        )
        np.testing.assert_array_equal(data.labels, labels_before)
        for s in data.flux:
            np.testing.assert_array_equal(data.flux[s], flux_before[s])

    def test_checkpoint_roundtrip(self, tmp_path):
        trainer = Trainer(make_config(tmp_path, epochs=2))
        train_idx, val_idx = split()
        trainer.fit_multi_survey_sparse(make_sparse_data(), train_idx, val_idx)
        path = trainer.save_checkpoint(tmp_path / "ckpt")
        for name in [
            "best_model.pth",
            "final_model.pth",
            "normalizer.pkl",
            "history_train_val.pkl",
            "learning_rates.pkl",
            "config.yaml",
        ]:
            assert (path / name).exists(), name
        config = ExperimentConfig.load_checkpoint_config(path / "config.yaml")
        assert config.parameter_names == PARAMS
        restored = Trainer(config).model
        restored.load_state_dict(torch.load(path / "best_model.pth", weights_only=True))

    @pytest.mark.parametrize(
        "scheduler",
        [
            SchedulerConfig(type=SchedulerType.REDUCE_ON_PLATEAU, patience=1),
            SchedulerConfig(type=SchedulerType.COSINE_ANNEALING),
            SchedulerConfig(type=SchedulerType.NONE),
        ],
    )
    def test_schedulers(self, tmp_path, scheduler):
        trainer = Trainer(make_config(tmp_path, epochs=3, scheduler=scheduler))
        train_idx, val_idx = split()
        history = trainer.fit_multi_survey_sparse(
            make_sparse_data(), train_idx, val_idx
        )
        assert len(history.val_losses) == 3
        assert all(np.isfinite(history.val_losses))

    def test_early_stopping(self, tmp_path):
        trainer = Trainer(
            make_config(
                tmp_path, epochs=50, early_stopping_patience=2, learning_rate=1e-6
            )
        )
        train_idx, val_idx = split()
        history = trainer.fit_multi_survey_sparse(
            make_sparse_data(), train_idx, val_idx
        )
        assert len(history.val_losses) < 50


class TestSparseMultiLabel:
    def test_per_labelset_history(self, tmp_path):
        trainer = Trainer(make_config(tmp_path, multi_label=True))
        train_idx, val_idx = split()
        history = trainer.fit_multi_survey_sparse(
            make_sparse_data(multi_label=True), train_idx, val_idx
        )
        assert history.label_source_names == ["apogee", "galah"]
        for source in ["apogee", "galah"]:
            losses = history.per_labelset_val_losses[source]
            assert len(losses) == 4 and all(np.isfinite(losses))
            assert len(history.per_labelset_val_metrics[source]["rmse"]) == 4
        assert all(np.isfinite(history.val_losses))

    def test_both_heads_learn(self, tmp_path):
        trainer = Trainer(
            make_config(tmp_path, multi_label=True, epochs=15, learning_rate=3e-3)
        )
        train_idx, val_idx = split()
        history = trainer.fit_multi_survey_sparse(
            make_sparse_data(multi_label=True), train_idx, val_idx
        )
        for source in ["apogee", "galah"]:
            losses = history.per_labelset_val_losses[source]
            assert min(losses) < losses[0], source

    def test_model_selection_uses_all_heads(self, tmp_path):
        """Validation loss must include the secondary head.

        Shifting GALAH labels on validation stars only leaves training
        unchanged, so the validation loss differs only if GALAH counts.
        """
        train_idx, val_idx = split()
        config = make_config(tmp_path, multi_label=True, epochs=2)
        h1 = Trainer(config).fit_multi_survey_sparse(
            make_sparse_data(multi_label=True), train_idx, val_idx
        )
        data2 = make_sparse_data(multi_label=True)
        data2.labels_dict["galah"][val_idx, 0, 2] += 0.5  # GALAH [Fe/H], val only
        h2 = Trainer(config).fit_multi_survey_sparse(data2, train_idx, val_idx)
        np.testing.assert_allclose(h1.train_losses, h2.train_losses)
        assert not np.allclose(h1.val_losses, h2.val_losses)

    def test_mismatched_label_sources_raise(self, tmp_path):
        trainer = Trainer(make_config(tmp_path, multi_label=True))
        train_idx, val_idx = split()
        with pytest.raises(ValueError, match="label sources"):
            trainer.fit_multi_survey_sparse(make_sparse_data(), train_idx, val_idx)


class TestMaskedLabels:
    """Masked label entries (e.g. stored as 0) must never produce NaN losses."""

    def test_single_survey_masked_teff(self, tmp_path):
        from dorothy.config.schema import ModelConfig

        rng = np.random.default_rng(0)
        n, n_wave = 80, 30
        latent = rng.normal(size=(n, 3)).astype(np.float32)
        X = np.zeros((n, 3, n_wave), dtype=np.float32)
        X[:, 0] = latent @ rng.normal(size=(3, n_wave)).astype(np.float32)
        X[:, 1] = 0.1
        X[:, 2] = 1.0
        y = _labels(rng, latent)
        y[:10, 0, 0] = 0.0  # masked Teff stored as 0 -> log10(0) = -inf
        y[:10, 1, 0] = 0.0
        y[:10, 2, 0] = 0.0
        config = ExperimentConfig(
            name="masked",
            output_dir=tmp_path,
            data=DataConfig(catalogue_path=tmp_path / "c.h5", parameter_names=PARAMS),
            model=ModelConfig(input_features=3 * n_wave, hidden_layers=[32, 16]),
            training=TrainingConfig(epochs=2, batch_size=16),
            device="cpu",
        )
        history = Trainer(config).fit(X[:60], y[:60], X[60:], y[60:])
        assert all(np.isfinite(history.train_losses))
        assert all(np.isfinite(history.val_losses))


class TestLabelSourceWeighting:
    """How losses from several label sources are combined."""

    @staticmethod
    def _outputs_targets(n_a=10, n_b=2, offset_b=1.0):
        torch.manual_seed(0)
        outputs = {k: torch.randn(12, 2, N_PARAMS) for k in ["apogee", "galah"]}
        targets = {}
        for k, n_valid, offset in [("apogee", n_a, 0.0), ("galah", n_b, offset_b)]:
            t = torch.zeros(12, 3, N_PARAMS)
            t[:, 0] = torch.randn(12, N_PARAMS) + offset
            t[:, 1] = 0.1
            t[:n_valid, 2] = 1.0
            targets[k] = t
        return outputs, targets

    def _trainer(self, tmp_path, **training):
        return Trainer(make_config(tmp_path, multi_label=True, epochs=1, **training))

    def test_per_label_pools_all_labels(self, tmp_path):
        trainer = self._trainer(tmp_path)
        outputs, targets = self._outputs_targets()
        elem = {
            k: trainer.loss_fn.element_losses(outputs[k], targets[k]) for k in targets
        }
        expected = (elem["apogee"].sum() + elem["galah"].sum()) / (12 * N_PARAMS)
        assert torch.allclose(trainer._combine_losses(outputs, targets), expected)

    def test_per_source_averages_heads(self, tmp_path):
        trainer = self._trainer(tmp_path, label_source_weighting="per_source")
        outputs, targets = self._outputs_targets()
        means = [trainer.loss_fn(outputs[k], targets[k]) for k in targets]
        expected = (means[0] + means[1]) / 2
        assert torch.allclose(trainer._combine_losses(outputs, targets), expected)

    def test_per_source_skips_sources_without_labels(self, tmp_path):
        trainer = self._trainer(tmp_path, label_source_weighting="per_source")
        outputs, targets = self._outputs_targets(n_b=0)
        expected = trainer.loss_fn(outputs["apogee"], targets["apogee"])
        assert torch.allclose(trainer._combine_losses(outputs, targets), expected)

    @pytest.mark.parametrize("weighting", ["per_label", "per_source"])
    def test_source_weights(self, tmp_path, weighting):
        trainer = self._trainer(
            tmp_path,
            label_source_weighting=weighting,
            label_source_weights={"galah": 0.0},
        )
        outputs, targets = self._outputs_targets()
        expected = trainer.loss_fn(outputs["apogee"], targets["apogee"])
        assert torch.allclose(trainer._combine_losses(outputs, targets), expected)

    def test_single_source_matches_loss_fn(self, tmp_path):
        trainer = Trainer(make_config(tmp_path, epochs=1))
        outputs, targets = self._outputs_targets()
        out = {"default": outputs["apogee"]}
        tgt = {"default": targets["apogee"]}
        assert torch.allclose(
            trainer._combine_losses(out, tgt),
            trainer.loss_fn(out["default"], tgt["default"]),
        )


class TestSharedLoopBehaviour:
    def test_per_labelset_best_epochs(self, tmp_path):
        trainer = Trainer(make_config(tmp_path, multi_label=True, epochs=5))
        train_idx, val_idx = split()
        history = trainer.fit_multi_survey_sparse(
            make_sparse_data(multi_label=True), train_idx, val_idx
        )
        for source in ["apogee", "galah"]:
            losses = history.per_labelset_val_losses[source]
            assert history.per_labelset_best_epoch[source] == int(np.argmin(losses))
            assert history.per_labelset_best_val_loss[source] == min(losses)

    def test_learning_rate_logged_per_step(self, tmp_path):
        trainer = Trainer(make_config(tmp_path, epochs=3))
        train_idx, val_idx = split()
        history = trainer.fit_multi_survey_sparse(
            make_sparse_data(), train_idx, val_idx
        )
        steps_per_epoch = -(-len(train_idx) // 32)
        assert len(history.learning_rates) == 3 * steps_per_epoch

    def test_history_pickle_has_best_epoch(self, tmp_path):
        import pickle

        trainer = Trainer(make_config(tmp_path, epochs=2))
        train_idx, val_idx = split()
        history = trainer.fit_multi_survey_sparse(
            make_sparse_data(), train_idx, val_idx
        )
        history.save(tmp_path / "h.pkl")
        with open(tmp_path / "h.pkl", "rb") as f:
            saved = pickle.load(f)
        assert saved["best_epoch"] == history.best_epoch
        assert saved["best_val_loss"] == history.best_val_loss
        assert saved["history_val"] == history.val_losses

    def test_mse_loss_trains(self, tmp_path):
        from dorothy.config.schema import LossType

        trainer = Trainer(make_config(tmp_path, epochs=3, loss=LossType.MSE))
        train_idx, val_idx = split()
        history = trainer.fit_multi_survey_sparse(
            make_sparse_data(), train_idx, val_idx
        )
        assert all(np.isfinite(history.train_losses))
        assert np.isfinite(history.val_metrics["rmse"][-1]).all()
        assert np.isnan(history.val_metrics["z_median"][-1]).all()

    def test_fit_multi_labelset_rejects_mismatched_sources(self, tmp_path):
        trainer = Trainer(make_config(tmp_path, multi_label=True))
        X = {s: np.zeros((10, 3, n), np.float32) for s, n in WAVES.items()}
        has = {s: np.ones(10, bool) for s in WAVES}
        y = {"apogee": np.zeros((10, 3, N_PARAMS), np.float32)}
        with pytest.raises(ValueError, match="label sources"):
            trainer.fit_multi_labelset(X, y, X, y, has, has)
