"""Tests for the catalogue loader module."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from dorothy.data.catalogue_loader import (
    PARAMETER_NAMES,
    CatalogueData,
    CatalogueInfo,
    CatalogueLoader,
)


class TestParameterNames:
    """Tests for parameter name constants."""

    def test_eleven_parameters(self):
        """Should have exactly 11 parameters."""
        assert len(PARAMETER_NAMES) == 11

    def test_expected_parameters(self):
        """Should contain expected parameter names."""
        expected = ["teff", "logg", "fe_h", "mg_fe", "c_fe"]
        for param in expected:
            assert param in PARAMETER_NAMES


class TestCatalogueData:
    """Tests for the CatalogueData dataclass."""

    @pytest.fixture
    def sample_data(self):
        """Create sample catalogue data."""
        n_stars = 100
        n_wave = 7650
        return CatalogueData(
            gaia_ids=np.arange(n_stars, dtype=np.int64),
            ra=np.random.uniform(0, 360, n_stars),
            dec=np.random.uniform(-90, 90, n_stars),
            flux=np.random.randn(n_stars, n_wave).astype(np.float32),
            ivar=np.abs(np.random.randn(n_stars, n_wave)).astype(np.float32),
            wavelength=np.linspace(3600, 9800, n_wave).astype(np.float32),
            snr=np.random.uniform(10, 100, n_stars).astype(np.float32),
            labels=np.random.randn(n_stars, 11).astype(np.float32),
            label_errors=np.abs(np.random.randn(n_stars, 11)).astype(np.float32),
            label_flags=np.zeros((n_stars, 11), dtype=np.uint8),
            survey_name="desi",
            label_source="apogee",
        )

    def test_n_stars(self, sample_data):
        """Should return correct number of stars."""
        assert sample_data.n_stars == 100

    def test_has_spectrum_computed(self, sample_data):
        """Should compute has_spectrum from ivar."""
        assert sample_data.has_spectrum.shape == (100,)
        assert sample_data.has_spectrum.dtype == bool
        assert np.all(sample_data.has_spectrum)  # All have positive ivar

    def test_has_labels_computed(self, sample_data):
        """Should compute has_labels from errors."""
        assert sample_data.has_labels.shape == (100,)
        assert sample_data.has_labels.dtype == bool
        assert np.all(sample_data.has_labels)  # All have positive errors

    def test_n_with_spectra(self, sample_data):
        """Should count stars with spectra."""
        assert sample_data.n_with_spectra == 100

    def test_n_with_labels(self, sample_data):
        """Should count stars with labels."""
        assert sample_data.n_with_labels == 100

    def test_n_complete(self, sample_data):
        """Should count complete stars."""
        assert sample_data.n_complete == 100

    def test_filter_complete(self, sample_data):
        """Should filter to complete stars only."""
        # Set some stars to have no spectra
        sample_data.ivar[50:] = 0
        sample_data.has_spectrum = np.any(sample_data.ivar > 0, axis=1)

        filtered = sample_data.filter_complete()
        assert filtered.n_stars == 50
        assert np.all(filtered.has_spectrum)
        assert np.all(filtered.has_labels)

    def test_filter_by_flags_no_flags(self, sample_data):
        """Should filter stars with no flags."""
        # Set some flags
        sample_data.label_flags[50:, 0] = 1  # Set bit 0 for half the stars

        filtered = sample_data.filter_by_flags(max_flag_bits=0)
        assert filtered.n_stars == 50

    def test_filter_by_flags_some_flags_allowed(self, sample_data):
        """Should allow stars with some flags when max_flag_bits > 0."""
        sample_data.label_flags[50:, 0] = 1  # 1 bit set
        sample_data.label_flags[75:, 1] = 1  # 2 bits set for some

        filtered = sample_data.filter_by_flags(max_flag_bits=1)
        # Stars with 0 or 1 bit set should pass
        assert filtered.n_stars > 50

    def test_missing_spectra_detected(self):
        """Should detect missing spectra from zero ivar."""
        n_stars = 10
        n_wave = 100
        data = CatalogueData(
            gaia_ids=np.arange(n_stars, dtype=np.int64),
            ra=np.zeros(n_stars),
            dec=np.zeros(n_stars),
            flux=np.zeros((n_stars, n_wave), dtype=np.float32),
            ivar=np.zeros((n_stars, n_wave), dtype=np.float32),  # All missing
            wavelength=np.linspace(3600, 9800, n_wave).astype(np.float32),
            snr=np.zeros(n_stars, dtype=np.float32),
            labels=np.zeros((n_stars, 11), dtype=np.float32),
            label_errors=np.zeros((n_stars, 11), dtype=np.float32),  # All missing
            label_flags=np.zeros((n_stars, 11), dtype=np.uint8),
            survey_name="test",
            label_source="test",
        )

        assert data.n_with_spectra == 0
        assert data.n_with_labels == 0
        assert data.n_complete == 0


class TestCatalogueLoader:
    """Tests for the CatalogueLoader class."""

    @pytest.fixture
    def mock_catalogue(self, tmp_path):
        """Create a mock HDF5 catalogue file."""
        path = tmp_path / "test_catalogue.h5"
        n_stars = 50
        n_wave_desi = 7650
        n_wave_boss = 4506

        with h5py.File(path, "w") as f:
            # Metadata
            meta = f.create_group("metadata")
            meta.create_dataset("gaia_id", data=np.arange(n_stars, dtype=np.int64))
            meta.create_dataset("ra", data=np.random.uniform(0, 360, n_stars))
            meta.create_dataset("dec", data=np.random.uniform(-90, 90, n_stars))

            # Surveys
            surveys = f.create_group("surveys")

            desi = surveys.create_group("desi")
            desi.create_dataset(
                "flux", data=np.random.randn(n_stars, n_wave_desi).astype(np.float32)
            )
            desi.create_dataset(
                "ivar",
                data=np.abs(np.random.randn(n_stars, n_wave_desi)).astype(np.float32),
            )
            desi.create_dataset(
                "wavelength",
                data=np.linspace(3600, 9800, n_wave_desi).astype(np.float32),
            )
            desi.create_dataset(
                "snr", data=np.random.uniform(20, 100, n_stars).astype(np.float32)
            )

            boss = surveys.create_group("boss")
            boss.create_dataset(
                "flux", data=np.random.randn(n_stars, n_wave_boss).astype(np.float32)
            )
            boss.create_dataset(
                "ivar",
                data=np.abs(np.random.randn(n_stars, n_wave_boss)).astype(np.float32),
            )
            boss.create_dataset(
                "wavelength",
                data=np.linspace(3600, 10400, n_wave_boss).astype(np.float32),
            )
            boss.create_dataset(
                "snr", data=np.random.uniform(20, 100, n_stars).astype(np.float32)
            )

            # Labels
            labels = f.create_group("labels")

            apogee = labels.create_group("apogee")
            apogee.create_dataset(
                "values", data=np.random.randn(n_stars, 11).astype(np.float32)
            )
            apogee.create_dataset(
                "errors", data=np.abs(np.random.randn(n_stars, 11)).astype(np.float32)
            )
            apogee.create_dataset("flags", data=np.zeros((n_stars, 11), dtype=np.uint8))

            galah = labels.create_group("galah")
            galah.create_dataset(
                "values", data=np.random.randn(n_stars, 11).astype(np.float32)
            )
            galah.create_dataset(
                "errors", data=np.abs(np.random.randn(n_stars, 11)).astype(np.float32)
            )
            galah.create_dataset("flags", data=np.zeros((n_stars, 11), dtype=np.uint8))

            # Attributes
            f.attrs["n_stars"] = n_stars
            f.attrs["parameter_names"] = PARAMETER_NAMES
            f.attrs["survey_names"] = ["desi", "boss", "lamost_lrs", "lamost_mrs"]
            f.attrs["creation_date"] = "2024-01-01T00:00:00"
            f.attrs["version"] = "1.0.0"

        return path

    def test_loader_init(self, mock_catalogue):
        """Should initialize with valid path."""
        loader = CatalogueLoader(mock_catalogue)
        assert loader.path == mock_catalogue

    def test_loader_init_missing_file(self, tmp_path):
        """Should raise error for missing file."""
        with pytest.raises(FileNotFoundError):
            CatalogueLoader(tmp_path / "nonexistent.h5")

    def test_get_info(self, mock_catalogue):
        """Should return catalogue info."""
        loader = CatalogueLoader(mock_catalogue)
        info = loader.get_info()

        assert isinstance(info, CatalogueInfo)
        assert info.n_stars == 50
        assert info.version == "1.0.0"
        assert "desi" in info.surveys
        assert "boss" in info.surveys
        assert "apogee" in info.labels
        assert "galah" in info.labels

    def test_get_info_survey_details(self, mock_catalogue):
        """Should return survey details."""
        loader = CatalogueLoader(mock_catalogue)
        info = loader.get_info()

        assert info.surveys["desi"]["wavelength_bins"] == 7650
        assert info.surveys["boss"]["wavelength_bins"] == 4506
        assert info.surveys["desi"]["n_with_data"] == 50

    def test_info_prints(self, mock_catalogue, capsys):
        """Should print info without error."""
        loader = CatalogueLoader(mock_catalogue)
        loader.info()
        captured = capsys.readouterr()
        assert "DOROTHY Super-Catalogue" in captured.out
        assert "desi" in captured.out

    def test_load_desi_apogee(self, mock_catalogue):
        """Should load DESI spectra with APOGEE labels."""
        loader = CatalogueLoader(mock_catalogue)
        data = loader.load(surveys="desi", label_source="apogee")

        assert data.n_stars == 50
        assert data.flux.shape == (50, 7650)
        assert data.labels.shape == (50, 11)
        assert data.survey_name == "desi"
        assert data.label_source == "apogee"

    def test_load_boss_galah(self, mock_catalogue):
        """Should load BOSS spectra with GALAH labels."""
        loader = CatalogueLoader(mock_catalogue)
        data = loader.load(surveys="boss", label_source="galah")

        assert data.flux.shape == (50, 4506)
        assert data.survey_name == "boss"
        assert data.label_source == "galah"

    def test_load_invalid_survey(self, mock_catalogue):
        """Should raise error for invalid survey."""
        loader = CatalogueLoader(mock_catalogue)
        with pytest.raises(ValueError, match="Survey 'invalid'"):
            loader.load(surveys="invalid", label_source="apogee")

    def test_load_invalid_label_source(self, mock_catalogue):
        """Should raise error for invalid label source."""
        loader = CatalogueLoader(mock_catalogue)
        with pytest.raises(ValueError, match="Label source 'invalid'"):
            loader.load(surveys="desi", label_source="invalid")

    def test_load_require_complete(self, mock_catalogue):
        """Should filter to complete stars when requested."""
        loader = CatalogueLoader(mock_catalogue)
        data = loader.load(surveys="desi", label_source="apogee", require_complete=True)

        assert data.n_stars == data.n_complete

    def test_load_default_survey(self, mock_catalogue):
        """Should load first available survey if none specified."""
        loader = CatalogueLoader(mock_catalogue)
        data = loader.load(label_source="apogee")

        # Should pick desi (first survey with data)
        assert data.survey_name in ["desi", "boss"]

    def test_load_for_training(self, mock_catalogue):
        """Should return training-ready arrays."""
        loader = CatalogueLoader(mock_catalogue)
        X, y, y_err, mask = loader.load_for_training(
            survey="desi", label_source="apogee"
        )

        assert X.shape == (50, 2, 7650)  # 2 channels: flux + ivar
        assert y.shape == (50, 11)
        assert y_err.shape == (50, 11)
        assert mask.shape == (50, 11)
        assert mask.dtype == bool

    def test_get_survey_wavelengths(self, mock_catalogue):
        """Should return wavelength grid for survey."""
        loader = CatalogueLoader(mock_catalogue)
        wavelengths = loader.get_survey_wavelengths("desi")

        assert wavelengths.shape == (7650,)
        assert wavelengths[0] < wavelengths[-1]  # Sorted

    def test_get_parameter_names(self, mock_catalogue):
        """Should return parameter names."""
        loader = CatalogueLoader(mock_catalogue)
        names = loader.get_parameter_names()

        assert len(names) == 11
        assert "teff" in names


class TestCatalogueLoaderMissingData:
    """Tests for handling missing data in catalogues."""

    @pytest.fixture
    def partial_catalogue(self, tmp_path):
        """Create a catalogue with partial data."""
        path = tmp_path / "partial_catalogue.h5"
        n_stars = 100

        with h5py.File(path, "w") as f:
            meta = f.create_group("metadata")
            meta.create_dataset("gaia_id", data=np.arange(n_stars, dtype=np.int64))
            meta.create_dataset("ra", data=np.zeros(n_stars))
            meta.create_dataset("dec", data=np.zeros(n_stars))

            surveys = f.create_group("surveys")
            desi = surveys.create_group("desi")

            # Only first 50 stars have DESI data
            flux = np.zeros((n_stars, 1000), dtype=np.float32)
            ivar = np.zeros((n_stars, 1000), dtype=np.float32)
            flux[:50] = np.random.randn(50, 1000)
            ivar[:50] = np.abs(np.random.randn(50, 1000))

            desi.create_dataset("flux", data=flux)
            desi.create_dataset("ivar", data=ivar)
            desi.create_dataset("wavelength", data=np.linspace(3600, 9800, 1000))
            desi.create_dataset(
                "snr", data=np.concatenate([np.ones(50) * 50, np.zeros(50)])
            )

            labels = f.create_group("labels")
            apogee = labels.create_group("apogee")

            # Only last 60 stars have APOGEE labels (30 overlap with DESI)
            values = np.zeros((n_stars, 11), dtype=np.float32)
            errors = np.zeros((n_stars, 11), dtype=np.float32)
            values[40:] = np.random.randn(60, 11)
            errors[40:] = np.abs(np.random.randn(60, 11))

            apogee.create_dataset("values", data=values)
            apogee.create_dataset("errors", data=errors)
            apogee.create_dataset("flags", data=np.zeros((n_stars, 11), dtype=np.uint8))

            f.attrs["n_stars"] = n_stars
            f.attrs["parameter_names"] = PARAMETER_NAMES
            f.attrs["survey_names"] = ["desi"]
            f.attrs["creation_date"] = "2024-01-01"
            f.attrs["version"] = "1.0.0"

        return path

    def test_partial_data_loaded(self, partial_catalogue):
        """Should load partial data correctly."""
        loader = CatalogueLoader(partial_catalogue)
        data = loader.load(surveys="desi", label_source="apogee")

        assert data.n_stars == 100
        assert data.n_with_spectra == 50  # First 50 have DESI
        assert data.n_with_labels == 60  # Last 60 have APOGEE
        assert data.n_complete == 10  # Overlap: stars 40-49

    def test_filter_complete_partial(self, partial_catalogue):
        """Should filter to complete stars correctly."""
        loader = CatalogueLoader(partial_catalogue)
        data = loader.load(surveys="desi", label_source="apogee")
        complete = data.filter_complete()

        assert complete.n_stars == 10
        assert complete.n_complete == 10
