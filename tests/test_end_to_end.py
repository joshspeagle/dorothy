"""End-to-end tests: train -> checkpoint -> evaluate / predict / info.

These use a synthetic v2 catalogue (tests/synthetic.py) and exercise the same
code paths as the CLI, including LAMOST MRS dual-arm spectra and two label
sources.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from dorothy.cli.main import main
from dorothy.config.schema import ExperimentConfig
from dorothy.data import CatalogueLoader
from dorothy.data.training_data import (
    DataSplit,
    complete_config_from_catalogue,
    load_training_data,
)
from dorothy.inference import Predictor, evaluate_model, load_checkpoint
from dorothy.training import Trainer
from tests.synthetic import PARAMS, SURVEY_WAVES, build_catalogue, write_config

MULTI_SURVEYS = ["boss", "desi", "lamost_mrs"]


@pytest.fixture(scope="module")
def catalogue(tmp_path_factory):
    return build_catalogue(tmp_path_factory.mktemp("cat") / "catalogue.h5")


def _config(tmp_path, catalogue, surveys, sources, **kw):
    path = write_config(
        tmp_path / "config.yaml",
        catalogue,
        tmp_path / "outputs",
        surveys,
        sources,
        **kw,
    )
    return path


def _train(config_path):
    """Train in-process like `dorothy train`, returning (config, data, split, trainer)."""
    import yaml

    raw = complete_config_from_catalogue(yaml.safe_load(config_path.read_text()))
    config = ExperimentConfig(**raw)
    loader = CatalogueLoader(config.data.catalogue_path)
    data = load_training_data(config, loader, verbose=False)
    data.configure_model(config)
    split = data.split(config.data.train_ratio, config.data.val_ratio, config.seed)
    trainer = Trainer(config)
    data.fit(trainer, split)
    return config, data, split, trainer


class TestEvaluateMatchesTraining:
    """`evaluate` on the validation split must reproduce training validation."""

    @pytest.mark.parametrize(
        "surveys,sources,extra",
        [
            (["boss"], ["apogee"], {}),
            (["boss"], ["apogee", "galah"], {}),
            (MULTI_SURVEYS, ["apogee"], {}),
            (MULTI_SURVEYS, ["apogee", "galah"], {}),
            (
                MULTI_SURVEYS,
                ["apogee", "galah"],
                {"extra_data": "  use_dense_loading: true\n"},
            ),
            (
                MULTI_SURVEYS,
                ["apogee", "galah"],
                {"extra_training": "  ema_decay: 0.9\n"},
            ),
        ],
        ids=[
            "single",
            "single_survey_multilabel",
            "sparse",
            "sparse_multilabel",
            "dense_multilabel",
            "weight_average",
        ],
    )
    def test_validation_metrics_reproduced(
        self, tmp_path, catalogue, surveys, sources, extra
    ):
        config_path = _config(tmp_path, catalogue, surveys, sources, epochs=3, **extra)
        config, data, split, trainer = _train(config_path)
        history = trainer.history
        path = trainer.save_checkpoint(tmp_path / "ckpt")

        ckpt = load_checkpoint(path, device="cpu")
        results = evaluate_model(
            ckpt.model, ckpt.normalizer, ckpt.config, data, split.val, device="cpu"
        )
        best = history.best_epoch
        primary = next(iter(results.values()))
        for metric in ["rmse", "bias", "z_robust_scatter"]:
            expected = history.val_metrics[metric][best]
            got = np.array(
                [getattr(primary.normalized.metrics[p], metric) for p in PARAMS]
            )
            np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-6)
        assert list(results) == sources  # heads named by label source
        if len(sources) > 1:
            for source in sources:
                expected = history.per_labelset_val_metrics[source]["rmse"][best]
                got = np.array(
                    [results[source].normalized.metrics[p].rmse for p in PARAMS]
                )
                np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-6)
        # Physical-space predictions are in label units (Teff in K)
        assert primary.physical is not None
        assert 3000 < np.nanmedian(primary.y_pred[:, 0]) < 8000

    @pytest.mark.parametrize(
        "surveys,sources",
        [(["boss"], ["apogee"]), (MULTI_SURVEYS, ["apogee", "galah"])],
        ids=["single", "sparse_multilabel"],
    )
    def test_calibrated_validation_z_scores_have_unit_spread(
        self, tmp_path, catalogue, surveys, sources
    ):
        config_path = _config(tmp_path, catalogue, surveys, sources, epochs=3)
        _, data, split, trainer = _train(config_path)
        ckpt = load_checkpoint(trainer.save_checkpoint(tmp_path / "ckpt"), device="cpu")
        results = evaluate_model(
            ckpt.model,
            ckpt.normalizer,
            ckpt.config,
            data,
            split.val,
            device="cpu",
            calibration=ckpt.calibration,
        )

        assert list(results) == list(ckpt.calibration) == sources
        for head in results.values():
            spread = [head.normalized.metrics[p].z_robust_scatter for p in PARAMS]
            # The tolerance of the reproduction above; measured within 2e-7
            np.testing.assert_allclose(spread, 1.0, rtol=1e-4)


class TestCLIEndToEnd:
    def test_train_evaluate_info_multilabel(self, tmp_path, catalogue, capsys):
        config_path = _config(tmp_path, catalogue, MULTI_SURVEYS, ["apogee", "galah"])
        assert main(["train", str(config_path)]) == 0
        ckpt = tmp_path / "outputs" / "e2e" / "e2e_final"
        for name in [
            "best_model.pth",
            "calibration.json",
            "config.yaml",
            "normalizer.pkl",
            "data_split.pkl",
        ]:
            assert (ckpt / name).exists(), name

        out = tmp_path / "results.json"
        assert (
            main(["evaluate", str(ckpt), "--format", "json", "--output", str(out)]) == 0
        )
        results = json.loads(out.read_text())
        assert set(results["label_sources"]) == {"apogee", "galah"}
        for head in results["label_sources"].values():
            assert "rmse" in head["normalized_space"]["fe_h"]
            assert "rmse" in head["physical_space"]["teff"]
        assert (
            results["normalized_space"]
            == results["label_sources"]["apogee"]["normalized_space"]
        )
        plots = ckpt / "plots" / "evaluation"
        for source in ["apogee", "galah"]:
            assert (plots / f"{source}_pred_vs_truth.png").exists()
            assert (plots / f"{source}_zscore_histograms.png").exists()

        capsys.readouterr()
        assert main(["info", str(ckpt)]) == 0
        info = capsys.readouterr().out
        assert "MultiHeadMLP" in info and "galah" in info
        assert "Parameters (4)" in info
        assert "Uncertainty model: label-aware" in info
        assert "Precision encoding: True" in info
        # One column of calibration factors per head
        table = info.split("Calibration factors of best_model.pth")[1]
        assert re.search(
            r"Parameter\s+apogee\s+galah\n\s+teff\s+[\d.]+\s+[\d.]+", table
        )

    def test_saved_split_matches_recreated_split(self, tmp_path, catalogue):
        config_path = _config(tmp_path, catalogue, MULTI_SURVEYS, ["apogee"])
        config, data, split, trainer = _train(config_path)
        path = trainer.save_checkpoint(tmp_path / "ckpt")
        split.save(path, data)
        loaded = DataSplit.load(path, data)
        for a, b in [(split.train, loaded.train), (split.test, loaded.test)]:
            np.testing.assert_array_equal(a, b)

    def test_resume_initializes_weights(self, tmp_path, catalogue, capsys):
        config_path = _config(tmp_path, catalogue, ["boss"], ["apogee"], epochs=1)
        assert main(["train", str(config_path)]) == 0
        ckpt = tmp_path / "outputs" / "e2e" / "e2e_final"
        assert main(["train", str(config_path), "--resume", str(ckpt)]) == 0
        assert "Initialized weights from" in capsys.readouterr().out

        # A checkpoint saved before the precision encoding was trained on sigma
        saved = ckpt / "config.yaml"
        text, removed = re.subn(
            r"^ *precision_encoding: true\n", "", saved.read_text(), flags=re.M
        )
        assert removed == 1
        saved.write_text(text)
        assert main(["train", str(config_path), "--resume", str(ckpt)]) == 1
        assert "before the precision encoding" in capsys.readouterr().out

    def test_resume_rejects_another_residual_stream_setting(
        self, tmp_path, catalogue, capsys
    ):
        config_path = _config(tmp_path, catalogue, MULTI_SURVEYS, ["apogee"], epochs=1)
        assert main(["train", str(config_path)]) == 0
        ckpt = tmp_path / "outputs" / "e2e" / "e2e_final"
        # A checkpoint saved before the residual stream has no residual_stream key
        saved = ckpt / "config.yaml"
        text, removed = re.subn(
            r"^ *residual_stream: true\n", "", saved.read_text(), flags=re.M
        )
        assert removed == 1
        saved.write_text(text)
        capsys.readouterr()
        assert main(["train", str(config_path), "--resume", str(ckpt)]) == 1
        assert "residual_stream: false" in capsys.readouterr().out


class TestPredictorMultiHead:
    def test_load_and_predict_single_survey_and_dict(self, tmp_path, catalogue):
        config_path = _config(tmp_path, catalogue, MULTI_SURVEYS, ["apogee", "galah"])
        config, data, split, trainer = _train(config_path)
        path = trainer.save_checkpoint(tmp_path / "ckpt")
        loader = CatalogueLoader(catalogue)

        star = int(np.where(data.sparse.has_data("lamost_mrs"))[0][0])
        X, _, _ = loader.load_spectrum("lamost_mrs", star, normalize_spectra=True)
        predictor = Predictor.load(path, device="cpu", survey="lamost_mrs")
        assert predictor.scatter_floor == config.training.scatter_floor
        result = predictor.predict(X[None], denormalize=True)
        assert result.predictions.shape == (1, len(PARAMS))
        assert not result.is_normalized

        # GALAH head differs from the APOGEE head
        galah = Predictor.load(
            path, device="cpu", survey="lamost_mrs", label_source="galah"
        )
        assert not np.allclose(galah.predict(X[None]).raw_output, result.raw_output)

        # Dict input with has_data reproduces the single-survey call
        X_dict = (
            {
                s: np.zeros((1, 3, n), np.float32)
                for s, n in trainer.model.survey_configs.items()
            }
            if hasattr(trainer.model, "survey_configs")
            else None
        )
        if X_dict is None:
            X_dict = {
                s: np.zeros((1, 3, enc.n_wavelengths), np.float32)
                for s, enc in trainer.model.encoders.items()
            }
        X_dict["lamost_mrs"] = X[None]
        has = {s: np.array([s == "lamost_mrs"]) for s in X_dict}
        dict_result = Predictor.load(path, device="cpu").predict(X_dict, has_data=has)
        np.testing.assert_allclose(dict_result.raw_output, result.raw_output, rtol=1e-5)

        # Trunk embeddings are available for multi-head models
        assert predictor.get_embeddings(X[None]).shape[0] == 1


def test_training_inputs_match_load_spectrum(tmp_path, catalogue):
    """load_training_data preprocesses spectra as load_spectrum does, so
    training, evaluation and prediction see the same inputs."""
    config_path = _config(tmp_path, catalogue, MULTI_SURVEYS, ["apogee"])
    import yaml

    raw = complete_config_from_catalogue(yaml.safe_load(config_path.read_text()))
    config = ExperimentConfig(**raw)
    assert config.data.normalize_spectra and config.data.precision_encoding
    loader = CatalogueLoader(catalogue)
    sparse = load_training_data(config, loader, verbose=False).sparse
    row = int(sparse.local_to_global["boss"][0])
    flux, sigma = sparse.flux["boss"][0], sparse.sigma["boss"][0]

    expected = loader.load_spectrum(
        "boss", row, normalize_spectra=True, precision_encoding=True
    )[0]
    np.testing.assert_array_equal(np.stack([flux, sigma, sigma > 0]), expected)


class TestPredictFromFits:
    @staticmethod
    def _write_fits(path, flux, ivar):
        from astropy.io import fits

        n, n_wave = flux.shape
        cols = [
            fits.Column(
                name="APOGEE_ID", format="20A", array=[f"star{i}" for i in range(n)]
            ),
            fits.Column(name="TEFF", format="D", array=np.full(n, 5000.0)),
        ]
        hdus = [
            fits.PrimaryHDU(),
            fits.BinTableHDU.from_columns(cols, name="APOGEE"),
            fits.BinTableHDU.from_columns(
                [fits.Column(name="FLUX", format=f"{n_wave}E", array=flux)],
                name="SPECTRA",
            ),
            fits.BinTableHDU.from_columns(
                [fits.Column(name="IVAR", format=f"{n_wave}E", array=ivar)],
                name="SPEC_IVAR",
            ),
        ]
        fits.HDUList(hdus).writeto(path)

    def test_predict_from_fits_matches_catalogue_predictions(self, tmp_path, catalogue):
        from dorothy.inference import predict_from_fits

        config_path = _config(tmp_path, catalogue, ["boss"], ["apogee"])
        config, data, split, trainer = _train(config_path)
        path = trainer.save_checkpoint(tmp_path / "ckpt")

        import h5py

        with h5py.File(catalogue, "r") as f:
            rows = np.where(f["surveys/boss/snr"][:] > 0)[0][:5]
            flux = f["surveys/boss/flux"][:][rows]
            ivar = f["surveys/boss/ivar"][:][rows]
        fits_path = tmp_path / "spectra.fits"
        self._write_fits(fits_path, flux, ivar)

        out = tmp_path / "pred.csv"
        assert predict_from_fits(fits_path, path, output_path=out, device="cpu") is None
        import pandas as pd

        df = pd.read_csv(out)
        assert len(df) == 5 and "teff" in df and "teff_err" in df

        # Same predictions as the catalogue path (training preprocessing)
        loader = CatalogueLoader(catalogue)
        X = np.stack(
            [
                loader.load_spectrum(
                    "boss", int(r), normalize_spectra=True, precision_encoding=True
                )[0]
                for r in rows
            ]
        )
        predictor = Predictor.load(path, device="cpu")
        expected = predictor.predict(X, denormalize=True)
        np.testing.assert_allclose(
            df["teff"].to_numpy(), expected.predictions[:, 0], rtol=1e-4
        )
        np.testing.assert_allclose(
            df["teff_err"].to_numpy(), expected.uncertainties[:, 0], rtol=1e-4
        )

    def test_predict_command_writes_calibrated_uncertainties_unless_told_not_to(
        self, tmp_path, catalogue
    ):
        import pandas as pd

        config_path = _config(tmp_path, catalogue, ["boss"], ["apogee"], epochs=1)
        _, _, _, trainer = _train(config_path)
        path = trainer.save_checkpoint(tmp_path / "ckpt")
        rng = np.random.default_rng(0)
        flux = rng.normal(10.0, 1.0, (4, SURVEY_WAVES["boss"])).astype(np.float32)
        fits_path = tmp_path / "spectra.fits"
        self._write_fits(fits_path, flux, np.full_like(flux, 50.0))

        tables = []
        for flags in ([], ["--no-calibration"]):
            out = tmp_path / f"predictions{len(flags)}.csv"
            args = ["--checkpoint", str(path), "--input", str(fits_path)]
            args += ["--output", str(out), "--device", "cpu", *flags]
            assert main(["predict", *args]) == 0
            tables.append(pd.read_csv(out))
        calibrated, raw = tables

        assert calibrated["calibrated"].all() and not raw["calibrated"].any()
        np.testing.assert_array_equal(calibrated["teff"], raw["teff"])
        for p in PARAMS:
            np.testing.assert_allclose(
                calibrated[f"{p}_err"] / raw[f"{p}_err"],
                trainer.calibration["apogee"][p],
                rtol=1e-6,  # float32 rounding of the scatter and the CSV
            )

    def test_wrong_wavelength_grid_raises(self, tmp_path, catalogue):
        from dorothy.inference import predict_from_fits

        config_path = _config(tmp_path, catalogue, ["boss"], ["apogee"], epochs=1)
        _, _, _, trainer = _train(config_path)
        path = trainer.save_checkpoint(tmp_path / "ckpt")
        fits_path = tmp_path / "bad.fits"
        self._write_fits(
            fits_path, np.ones((2, 7), np.float32), np.ones((2, 7), np.float32)
        )
        with pytest.raises(ValueError, match="wavelength bins"):
            predict_from_fits(fits_path, path, device="cpu")


def test_single_survey_multi_head_fit(tmp_path, catalogue):
    """A single-survey config with a multi_head_model trains via fit()."""
    extra = ""
    config_path = write_config(
        tmp_path / "config.yaml", catalogue, tmp_path / "out", ["boss"], ["apogee"]
    )
    text = config_path.read_text().replace(
        "model:\n  hidden_layers: [32, 16]\n",
        "multi_head_model:\n  latent_dim: 16\n  encoder_hidden: [16]\n"
        "  trunk_hidden: [16]\n  output_hidden: [8]\n",
    )
    config_path.write_text(text + extra)
    assert main(["train", str(config_path)]) == 0
