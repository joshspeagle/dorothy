"""Tests for config-driven data loading, filtering and splitting."""

from __future__ import annotations

import pickle

import h5py
import numpy as np
import pytest

from dorothy.config.schema import ExperimentConfig
from dorothy.data.training_data import DataSplit, TrainingData, load_training_data
from tests.synthetic import PARAMS, build_catalogue


@pytest.fixture(scope="module")
def catalogue(tmp_path_factory):
    return build_catalogue(tmp_path_factory.mktemp("cat") / "catalogue.h5")


def _config(catalogue, surveys, sources, **data):
    multi_head = None
    if len(surveys) > 1:
        from dorothy.data import CatalogueLoader

        multi_head = {
            "survey_wavelengths": CatalogueLoader(
                catalogue
            ).get_survey_wavelength_counts(surveys),
            "label_sources": sources if len(sources) > 1 else None,
        }
    return ExperimentConfig(
        name="t",
        data={
            "catalogue_path": catalogue,
            "surveys": surveys,
            "label_sources": sources,
            "parameter_names": PARAMS,
            "max_flag_bits": -1,
            **data,
        },
        multi_head_model=multi_head,
    )


class TestFiltering:
    def test_sparse_single_label_keeps_partially_labelled_stars(self, catalogue):
        config = _config(catalogue, ["boss", "desi"], ["apogee"])
        data = load_training_data(config, verbose=False)
        labels = data.sparse.labels
        # Mask out one parameter for some stars: they must still be valid
        has_label = (labels[:, 2, :] > 0).any(axis=1)
        has_survey = data.sparse.has_data("boss") | data.sparse.has_data("desi")
        np.testing.assert_array_equal(
            data.valid_indices, np.where(has_label & has_survey)[0]
        )

    def test_partial_labels_kept_after_masking_one_parameter(self, catalogue):
        config = _config(catalogue, ["boss", "desi"], ["apogee"])
        data = load_training_data(config, verbose=False)
        n_before = len(data.valid_indices)
        data.sparse.labels[:, 2, 3] = 0.0  # no star has a valid mg_h now
        from dorothy.data.training_data import _valid_rows

        has_survey = data.sparse.has_data("boss") | data.sparse.has_data("desi")
        assert len(_valid_rows(has_survey, [data.sparse.labels])) == n_before

    def test_multi_label_uses_any_source(self, catalogue):
        config = _config(catalogue, ["boss", "desi"], ["apogee", "galah"])
        data = load_training_data(config, verbose=False)
        any_label = np.zeros(data.n_rows, dtype=bool)
        for source in ["apogee", "galah"]:
            any_label |= (data.labels(source)[:, 2, :] > 0).any(axis=1)
        assert set(data.valid_indices) <= set(np.where(any_label)[0])
        assert len(data.valid_indices) > len(
            load_training_data(
                _config(catalogue, ["boss", "desi"], ["apogee"]), verbose=False
            ).valid_indices
        )

    def test_duplicate_labels(self, catalogue):
        config = _config(
            catalogue,
            ["boss", "desi"],
            ["apogee", "galah"],
            duplicate_labels={"galah": "apogee"},
        )
        data = load_training_data(config, verbose=False)
        np.testing.assert_array_equal(data.labels("galah"), data.labels("apogee"))


class TestSplits:
    def test_split_reproduces_previous_permutation(self, catalogue):
        config = _config(catalogue, ["boss"], ["apogee"])
        data = load_training_data(config, verbose=False)
        split = data.split(0.7, 0.2, seed=42)
        # Previous versions: permutation of the filtered rows with RandomState
        n = len(data.valid_indices)
        perm = np.random.RandomState(42).permutation(n)
        n_train, n_val = int(n * 0.7), int(n * 0.2)
        np.testing.assert_array_equal(split.train, data.valid_indices[perm[:n_train]])
        np.testing.assert_array_equal(
            split.test, data.valid_indices[perm[n_train + n_val :]]
        )
        assert len(set(split.train) & set(split.test)) == 0

    def test_save_load_roundtrip(self, tmp_path, catalogue):
        data = load_training_data(
            _config(catalogue, ["boss"], ["apogee"]), verbose=False
        )
        split = data.split(0.7, 0.2, seed=1)
        split.save(tmp_path, data)
        loaded = DataSplit.load(tmp_path, data)
        np.testing.assert_array_equal(split.val, loaded.val)
        np.testing.assert_array_equal(split.test, loaded.test)

    def test_legacy_single_survey_split_file(self, tmp_path, catalogue):
        """Old files stored indices into the filtered arrays."""
        data = load_training_data(
            _config(catalogue, ["boss"], ["apogee"]), verbose=False
        )
        filtered = data.legacy_filtered_rows()
        old_test = np.array([0, 3, 5])
        with open(tmp_path / "data_split.pkl", "wb") as f:
            pickle.dump(
                {
                    "train_idx": np.array([1, 2]),
                    "val_idx": np.array([4]),
                    "test_idx": old_test,
                    "split_mode": "single_survey",
                    "n_total": len(filtered),
                },
                f,
            )
        loaded = DataSplit.load(tmp_path, data)
        np.testing.assert_array_equal(loaded.test, filtered[old_test])

    def test_legacy_dense_rows_count_stars_with_any_positive_label_error(
        self, tmp_path
    ):
        """Old dense files indexed the stars with a survey and a positive
        label error, whether or not the flag filter kept a label."""
        path = build_catalogue(tmp_path / "cat.h5")
        with h5py.File(path, "r+") as f:
            has_survey = (f["surveys/boss/snr"][:] > 0) | (f["surveys/desi/snr"][:] > 0)
            has_error = np.zeros(len(has_survey), dtype=bool)
            for source in ("apogee", "galah"):
                f["labels"][source]["flags"][:20] = 1
                has_error |= (f["labels"][source]["errors"][:] > 0).any(axis=1)
        config = _config(
            path,
            ["boss", "desi"],
            ["apogee", "galah"],
            max_flag_bits=0,
            use_dense_loading=True,
        )
        data = load_training_data(config, verbose=False)
        # The loaded rows are the stars with a survey, in catalogue order
        legacy = data.legacy_filtered_rows()
        np.testing.assert_array_equal(legacy, np.flatnonzero(has_error[has_survey]))
        assert len(data.valid_indices) < len(legacy)

    def test_legacy_sparse_split_file_is_already_rows(self, tmp_path):
        data = TrainingData("sparse_multi_survey", ["apogee"], np.arange(10))
        with open(tmp_path / "data_split.pkl", "wb") as f:
            pickle.dump(
                {
                    "train_idx": np.array([1]),
                    "val_idx": np.array([2]),
                    "test_idx": np.array([7, 9]),
                    "split_mode": "sparse_multi_survey",
                    "n_total": 10,
                    "valid_indices": np.arange(10),
                },
                f,
            )
        np.testing.assert_array_equal(DataSplit.load(tmp_path, data).test, [7, 9])


class TestCompleteConfig:
    def test_fills_wavelengths_heads_and_parameters(self, catalogue):
        from dorothy.data.training_data import complete_config_from_catalogue

        raw = {
            "name": "t",
            "data": {
                "catalogue_path": str(catalogue),
                "surveys": ["boss", "lamost_mrs"],
                "label_sources": ["apogee", "galah"],
            },
        }
        done = complete_config_from_catalogue(raw)
        assert "multi_head_model" not in raw  # input not modified
        assert done["multi_head_model"]["survey_wavelengths"] == {
            "boss": 40,
            "lamost_mrs": 28,
        }
        assert done["multi_head_model"]["label_sources"] == ["apogee", "galah"]
        assert done["data"]["parameter_names"] == PARAMS
        assert ExperimentConfig(**done).multi_head_model.n_parameters == len(PARAMS)

    def test_mismatched_parameter_names_raise(self, catalogue):
        from dorothy.data.training_data import complete_config_from_catalogue

        raw = {
            "name": "t",
            "data": {"catalogue_path": str(catalogue), "parameter_names": ["teff"]},
        }
        with pytest.raises(ValueError, match="does not match"):
            complete_config_from_catalogue(raw)


def test_parameter_names_must_match_catalogue(catalogue):
    config = _config(catalogue, ["boss"], ["apogee"])
    config.data.parameter_names = ["teff", "logg", "fe_h", "mg_fe"]
    with pytest.raises(ValueError, match="do not match"):
        load_training_data(config, verbose=False)
