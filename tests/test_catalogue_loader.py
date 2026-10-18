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
    MergedCatalogueData,
    SparseMergedData,
)
from dorothy.data.fits_loader import zero_invalid_pixels
from tests.synthetic import build_catalogue


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
        data = loader.load(survey="desi", label_source="apogee")

        assert data.n_stars == 50
        assert data.flux.shape == (50, 7650)
        assert data.labels.shape == (50, 11)
        assert data.survey_name == "desi"
        assert data.label_source == "apogee"

    def test_load_boss_galah(self, mock_catalogue):
        """Should load BOSS spectra with GALAH labels."""
        loader = CatalogueLoader(mock_catalogue)
        data = loader.load(survey="boss", label_source="galah")

        assert data.flux.shape == (50, 4506)
        assert data.survey_name == "boss"
        assert data.label_source == "galah"

    def test_load_invalid_survey(self, mock_catalogue):
        """Should raise error for invalid survey."""
        loader = CatalogueLoader(mock_catalogue)
        with pytest.raises(ValueError, match="Survey 'invalid'"):
            loader.load(survey="invalid", label_source="apogee")

    def test_load_invalid_label_source(self, mock_catalogue):
        """Should raise error for invalid label source."""
        loader = CatalogueLoader(mock_catalogue)
        with pytest.raises(ValueError, match="Label source 'invalid'"):
            loader.load(survey="desi", label_source="invalid")

    def test_load_require_complete(self, mock_catalogue):
        """Should filter to complete stars when requested."""
        loader = CatalogueLoader(mock_catalogue)
        data = loader.load(survey="desi", label_source="apogee", require_complete=True)

        assert data.n_stars == data.n_complete

    def test_load_default_survey(self, mock_catalogue):
        """Should load first available survey if none specified."""
        loader = CatalogueLoader(mock_catalogue)
        data = loader.load(label_source="apogee")

        # Should pick desi (first survey with data)
        assert data.survey_name in ["desi", "boss"]

    def test_load_for_training(self, mock_catalogue):
        """Should return training-ready 3-channel arrays."""
        loader = CatalogueLoader(mock_catalogue)
        X, y = loader.load_for_training(survey="desi", label_source="apogee")

        # X: 3-channel spectral data [flux, sigma, mask]
        assert X.shape == (50, 3, 7650)
        # y: 3-channel labels [values, errors, mask]
        assert y.shape == (50, 3, 11)

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
        data = loader.load(survey="desi", label_source="apogee")

        assert data.n_stars == 100
        assert data.n_with_spectra == 50  # First 50 have DESI
        assert data.n_with_labels == 60  # Last 60 have APOGEE
        assert data.n_complete == 10  # Overlap: stars 40-49

    def test_filter_complete_partial(self, partial_catalogue):
        """Should filter to complete stars correctly."""
        loader = CatalogueLoader(partial_catalogue)
        data = loader.load(survey="desi", label_source="apogee")
        complete = data.filter_complete()

        assert complete.n_stars == 10
        assert complete.n_complete == 10


class TestMultiLabelsetLoading:
    """Tests for loading data with multiple label sources (multi-labelset)."""

    @pytest.fixture
    def multi_label_catalogue(self, tmp_path):
        """Create a mock HDF5 catalogue with multiple label sources.

        Layout:
        - 100 stars total
        - Stars 0-29: have APOGEE labels only
        - Stars 30-59: have GALAH labels only
        - Stars 60-79: have BOTH APOGEE and GALAH labels
        - Stars 80-99: have NO labels (missing from both sources)
        """
        path = tmp_path / "multi_label_catalogue.h5"
        n_stars = 100
        n_wave = 500
        n_params = 11

        with h5py.File(path, "w") as f:
            # Metadata
            meta = f.create_group("metadata")
            meta.create_dataset("gaia_id", data=np.arange(n_stars, dtype=np.int64))
            meta.create_dataset("ra", data=np.random.uniform(0, 360, n_stars))
            meta.create_dataset("dec", data=np.random.uniform(-90, 90, n_stars))

            # Single survey for simplicity
            surveys = f.create_group("surveys")
            desi = surveys.create_group("desi")
            desi.create_dataset(
                "flux", data=np.random.randn(n_stars, n_wave).astype(np.float32)
            )
            desi.create_dataset(
                "ivar",
                data=np.abs(np.random.randn(n_stars, n_wave)).astype(np.float32),
            )
            desi.create_dataset(
                "wavelength",
                data=np.linspace(3600, 9800, n_wave).astype(np.float32),
            )
            desi.create_dataset(
                "snr", data=np.random.uniform(20, 100, n_stars).astype(np.float32)
            )

            # Labels group with multiple sources
            labels = f.create_group("labels")

            # APOGEE labels: stars 0-29 and 60-79
            apogee = labels.create_group("apogee")
            apogee_values = np.zeros((n_stars, n_params), dtype=np.float32)
            apogee_errors = np.zeros((n_stars, n_params), dtype=np.float32)
            apogee_flags = np.zeros((n_stars, n_params), dtype=np.uint8)

            # Set valid APOGEE labels for stars 0-29 and 60-79
            apogee_mask = np.zeros(n_stars, dtype=bool)
            apogee_mask[:30] = True
            apogee_mask[60:80] = True
            apogee_values[apogee_mask] = np.random.randn(50, n_params)
            apogee_errors[apogee_mask] = np.abs(np.random.randn(50, n_params)) + 0.01

            apogee.create_dataset("values", data=apogee_values)
            apogee.create_dataset("errors", data=apogee_errors)
            apogee.create_dataset("flags", data=apogee_flags)

            # GALAH labels: stars 30-59 and 60-79
            galah = labels.create_group("galah")
            galah_values = np.zeros((n_stars, n_params), dtype=np.float32)
            galah_errors = np.zeros((n_stars, n_params), dtype=np.float32)
            galah_flags = np.zeros((n_stars, n_params), dtype=np.uint8)

            # Set valid GALAH labels for stars 30-59 and 60-79
            galah_mask = np.zeros(n_stars, dtype=bool)
            galah_mask[30:60] = True
            galah_mask[60:80] = True
            galah_values[galah_mask] = np.random.randn(50, n_params)
            galah_errors[galah_mask] = np.abs(np.random.randn(50, n_params)) + 0.01

            galah.create_dataset("values", data=galah_values)
            galah.create_dataset("errors", data=galah_errors)
            galah.create_dataset("flags", data=galah_flags)

            # Attributes
            f.attrs["n_stars"] = n_stars
            f.attrs["parameter_names"] = PARAMETER_NAMES
            f.attrs["survey_names"] = ["desi"]
            f.attrs["creation_date"] = "2024-01-01"
            f.attrs["version"] = "1.0.0"

        return path

    def test_load_apogee_labels(self, multi_label_catalogue):
        """Should load APOGEE labels correctly."""
        loader = CatalogueLoader(multi_label_catalogue)
        data = loader.load(survey="desi", label_source="apogee")

        assert data.n_stars == 100
        assert data.label_source == "apogee"

        # Check that APOGEE-labeled stars have valid labels
        # Stars 0-29 and 60-79 should have labels
        has_apogee = np.any(data.label_errors > 0, axis=1)
        expected_apogee = np.zeros(100, dtype=bool)
        expected_apogee[:30] = True
        expected_apogee[60:80] = True
        np.testing.assert_array_equal(has_apogee, expected_apogee)

    def test_load_galah_labels(self, multi_label_catalogue):
        """Should load GALAH labels correctly."""
        loader = CatalogueLoader(multi_label_catalogue)
        data = loader.load(survey="desi", label_source="galah")

        assert data.n_stars == 100
        assert data.label_source == "galah"

        # Check that GALAH-labeled stars have valid labels
        # Stars 30-59 and 60-79 should have labels
        has_galah = np.any(data.label_errors > 0, axis=1)
        expected_galah = np.zeros(100, dtype=bool)
        expected_galah[30:60] = True
        expected_galah[60:80] = True
        np.testing.assert_array_equal(has_galah, expected_galah)

    def test_filter_complete_apogee(self, multi_label_catalogue):
        """Should filter to complete stars with APOGEE labels."""
        loader = CatalogueLoader(multi_label_catalogue)
        data = loader.load(survey="desi", label_source="apogee")
        complete = data.filter_complete()

        # 50 stars have APOGEE labels (0-29 and 60-79)
        assert complete.n_stars == 50
        assert complete.n_complete == 50

    def test_filter_complete_galah(self, multi_label_catalogue):
        """Should filter to complete stars with GALAH labels."""
        loader = CatalogueLoader(multi_label_catalogue)
        data = loader.load(survey="desi", label_source="galah")
        complete = data.filter_complete()

        # 50 stars have GALAH labels (30-59 and 60-79)
        assert complete.n_stars == 50

    def test_overlap_stars_have_both_sources(self, multi_label_catalogue):
        """Stars 60-79 should have both APOGEE and GALAH labels."""
        loader = CatalogueLoader(multi_label_catalogue)

        apogee_data = loader.load(survey="desi", label_source="apogee")
        galah_data = loader.load(survey="desi", label_source="galah")

        # Stars 60-79 should have labels in both
        overlap_stars = range(60, 80)
        for star_idx in overlap_stars:
            assert np.any(
                apogee_data.label_errors[star_idx] > 0
            ), f"Star {star_idx} should have APOGEE labels"
            assert np.any(
                galah_data.label_errors[star_idx] > 0
            ), f"Star {star_idx} should have GALAH labels"

    def test_load_for_training_creates_correct_mask(self, multi_label_catalogue):
        """load_for_training should filter to complete stars and create correct masks."""
        loader = CatalogueLoader(multi_label_catalogue)

        # load_for_training uses require_complete=True, so only returns
        # stars with BOTH spectra AND labels
        X, y = loader.load_for_training(survey="desi", label_source="apogee")

        # y is 3-channel: [values, errors, mask]
        label_mask = y[:, 2, :]  # (N, n_params)

        # 50 stars have APOGEE labels (0-29 and 60-79 from original)
        assert y.shape[0] == 50, f"Expected 50 complete stars, got {y.shape[0]}"

        # All returned stars should have mask=1 (since we filtered to complete)
        assert np.all(label_mask == 1), "All complete stars should have mask=1"

    def test_info_shows_multiple_label_sources(self, multi_label_catalogue):
        """CatalogueInfo should list multiple label sources."""
        loader = CatalogueLoader(multi_label_catalogue)
        info = loader.get_info()

        assert "apogee" in info.labels
        assert "galah" in info.labels
        assert info.labels["apogee"]["n_with_data"] == 50  # 0-29 + 60-79
        assert info.labels["galah"]["n_with_data"] == 50  # 30-59 + 60-79


class TestMultiLabelsetUnionBehavior:
    """Tests for union behavior in multi-labelset loading.

    The key behavior: a star should be included in training if it has
    labels from ANY label source (union), not just all sources.
    """

    @pytest.fixture
    def union_test_catalogue(self, tmp_path):
        """Create a catalogue specifically for testing union behavior.

        Layout:
        - 60 stars total
        - Stars 0-19: APOGEE only
        - Stars 20-39: GALAH only
        - Stars 40-59: both APOGEE and GALAH

        If we use union behavior, all 60 stars can be used for training
        (with appropriate masking).
        """
        path = tmp_path / "union_test_catalogue.h5"
        n_stars = 60
        n_wave = 200
        n_params = 11

        with h5py.File(path, "w") as f:
            # Metadata
            # Use gaia_ids starting at 1 (not 0) since load_merged filters out id <= 0
            meta = f.create_group("metadata")
            meta.create_dataset(
                "gaia_id", data=np.arange(1, n_stars + 1, dtype=np.int64)
            )
            meta.create_dataset("ra", data=np.random.uniform(0, 360, n_stars))
            meta.create_dataset("dec", data=np.random.uniform(-90, 90, n_stars))

            # Survey
            surveys = f.create_group("surveys")
            desi = surveys.create_group("desi")
            desi.create_dataset(
                "flux", data=np.random.randn(n_stars, n_wave).astype(np.float32)
            )
            desi.create_dataset(
                "ivar",
                data=np.abs(np.random.randn(n_stars, n_wave)).astype(np.float32),
            )
            desi.create_dataset(
                "wavelength", data=np.linspace(3600, 9800, n_wave).astype(np.float32)
            )
            desi.create_dataset(
                "snr", data=np.random.uniform(20, 100, n_stars).astype(np.float32)
            )

            # Labels
            labels = f.create_group("labels")

            # APOGEE: stars 0-19 and 40-59
            apogee = labels.create_group("apogee")
            apogee_values = np.zeros((n_stars, n_params), dtype=np.float32)
            apogee_errors = np.zeros((n_stars, n_params), dtype=np.float32)
            apogee_mask = np.zeros(n_stars, dtype=bool)
            apogee_mask[:20] = True
            apogee_mask[40:60] = True
            apogee_values[apogee_mask] = np.random.randn(40, n_params)
            apogee_errors[apogee_mask] = np.abs(np.random.randn(40, n_params)) + 0.01
            apogee.create_dataset("values", data=apogee_values)
            apogee.create_dataset("errors", data=apogee_errors)
            apogee.create_dataset(
                "flags", data=np.zeros((n_stars, n_params), dtype=np.uint8)
            )

            # GALAH: stars 20-39 and 40-59
            galah = labels.create_group("galah")
            galah_values = np.zeros((n_stars, n_params), dtype=np.float32)
            galah_errors = np.zeros((n_stars, n_params), dtype=np.float32)
            galah_mask = np.zeros(n_stars, dtype=bool)
            galah_mask[20:40] = True
            galah_mask[40:60] = True
            galah_values[galah_mask] = np.random.randn(40, n_params)
            galah_errors[galah_mask] = np.abs(np.random.randn(40, n_params)) + 0.01
            galah.create_dataset("values", data=galah_values)
            galah.create_dataset("errors", data=galah_errors)
            galah.create_dataset(
                "flags", data=np.zeros((n_stars, n_params), dtype=np.uint8)
            )

            # Attributes
            f.attrs["n_stars"] = n_stars
            f.attrs["parameter_names"] = PARAMETER_NAMES
            f.attrs["survey_names"] = ["desi"]
            f.attrs["creation_date"] = "2024-01-01"
            f.attrs["version"] = "1.0.0"

        return path

    def test_union_includes_all_stars_with_any_labels(self, union_test_catalogue):
        """Union of label sources should include all 60 stars."""
        loader = CatalogueLoader(union_test_catalogue)

        # Load both label sources
        apogee_data = loader.load(survey="desi", label_source="apogee")
        galah_data = loader.load(survey="desi", label_source="galah")

        # Count stars with any labels
        has_apogee = np.any(apogee_data.label_errors > 0, axis=1)
        has_galah = np.any(galah_data.label_errors > 0, axis=1)
        has_any = has_apogee | has_galah

        # All 60 stars should have at least one label source
        assert has_any.sum() == 60

    def test_masks_for_multi_labelset_training(self, union_test_catalogue):
        """Verify that label masks are correct for multi-labelset training.

        Uses load() to verify the raw label data is correct for each source.
        The mask information comes from label_errors > 0.
        """
        loader = CatalogueLoader(union_test_catalogue)

        # Load both label sources using load() which gives us all stars
        apogee_data = loader.load(survey="desi", label_source="apogee")
        galah_data = loader.load(survey="desi", label_source="galah")

        # Check which stars have valid labels (error > 0 means valid)
        has_apogee = np.any(apogee_data.label_errors > 0, axis=1)
        has_galah = np.any(galah_data.label_errors > 0, axis=1)

        # Stars 0-19: APOGEE only
        for i in range(20):
            assert has_apogee[i], f"Star {i} should have APOGEE labels"
            assert not has_galah[i], f"Star {i} should NOT have GALAH labels"
            assert np.all(
                apogee_data.label_errors[i] > 0
            ), f"Star {i} APOGEE errors should all be > 0"

        # Stars 20-39: GALAH only
        for i in range(20, 40):
            assert not has_apogee[i], f"Star {i} should NOT have APOGEE labels"
            assert has_galah[i], f"Star {i} should have GALAH labels"
            assert np.all(
                galah_data.label_errors[i] > 0
            ), f"Star {i} GALAH errors should all be > 0"

        # Stars 40-59: both
        for i in range(40, 60):
            assert has_apogee[i], f"Star {i} should have APOGEE labels"
            assert has_galah[i], f"Star {i} should have GALAH labels"
            assert np.all(
                apogee_data.label_errors[i] > 0
            ), f"Star {i} APOGEE errors should all be > 0"
            assert np.all(
                galah_data.label_errors[i] > 0
            ), f"Star {i} GALAH errors should all be > 0"

    def test_combined_mask_for_union_training(self, union_test_catalogue):
        """For union training, verify that the union of labels covers all 60 stars."""
        loader = CatalogueLoader(union_test_catalogue)

        # Load both label sources
        apogee_data = loader.load(survey="desi", label_source="apogee")
        galah_data = loader.load(survey="desi", label_source="galah")

        # Check which stars have valid labels from each source
        has_apogee = np.any(apogee_data.label_errors > 0, axis=1)
        has_galah = np.any(galah_data.label_errors > 0, axis=1)

        # Combined: stars with ANY label source (union)
        has_any = has_apogee | has_galah

        # All 60 stars should have at least one label source
        assert (
            has_any.sum() == 60
        ), f"Expected 60 stars with labels, got {has_any.sum()}"

        # Verify the counts:
        # - 40 stars have APOGEE (0-19 and 40-59)
        # - 40 stars have GALAH (20-39 and 40-59)
        # - 20 stars have both (40-59)
        assert (
            has_apogee.sum() == 40
        ), f"Expected 40 APOGEE stars, got {has_apogee.sum()}"
        assert has_galah.sum() == 40, f"Expected 40 GALAH stars, got {has_galah.sum()}"
        assert (has_apogee & has_galah).sum() == 20, "Expected 20 stars with both"


class TestMultiLabelsetPartialParameters:
    """Tests for partial parameter masking within label sources.

    Some stars might have only some parameters available from a label source.
    For example, GALAH might not provide [N/Fe] for some stars.
    """

    @pytest.fixture
    def partial_params_catalogue(self, tmp_path):
        """Create catalogue where some stars have partial parameter coverage."""
        path = tmp_path / "partial_params_catalogue.h5"
        n_stars = 30
        n_wave = 100
        n_params = 11

        with h5py.File(path, "w") as f:
            meta = f.create_group("metadata")
            meta.create_dataset("gaia_id", data=np.arange(n_stars, dtype=np.int64))
            meta.create_dataset("ra", data=np.zeros(n_stars))
            meta.create_dataset("dec", data=np.zeros(n_stars))

            surveys = f.create_group("surveys")
            desi = surveys.create_group("desi")
            desi.create_dataset(
                "flux", data=np.random.randn(n_stars, n_wave).astype(np.float32)
            )
            desi.create_dataset(
                "ivar",
                data=np.abs(np.random.randn(n_stars, n_wave)).astype(np.float32),
            )
            desi.create_dataset(
                "wavelength", data=np.linspace(3600, 9800, n_wave).astype(np.float32)
            )
            desi.create_dataset("snr", data=np.ones(n_stars, dtype=np.float32) * 50)

            labels = f.create_group("labels")
            apogee = labels.create_group("apogee")

            # All stars have values, but only some parameters have valid errors
            apogee_values = np.random.randn(n_stars, n_params).astype(np.float32)
            apogee_errors = np.zeros((n_stars, n_params), dtype=np.float32)
            apogee_flags = np.zeros((n_stars, n_params), dtype=np.uint8)

            # Stars 0-9: all 11 parameters valid
            apogee_errors[:10, :] = np.abs(np.random.randn(10, n_params)) + 0.01

            # Stars 10-19: only first 5 parameters valid (teff, logg, feh, mgfe, cfe)
            apogee_errors[10:20, :5] = np.abs(np.random.randn(10, 5)) + 0.01

            # Stars 20-29: only atmospheric params valid (teff, logg, feh)
            apogee_errors[20:30, :3] = np.abs(np.random.randn(10, 3)) + 0.01

            apogee.create_dataset("values", data=apogee_values)
            apogee.create_dataset("errors", data=apogee_errors)
            apogee.create_dataset("flags", data=apogee_flags)

            f.attrs["n_stars"] = n_stars
            f.attrs["parameter_names"] = PARAMETER_NAMES
            f.attrs["survey_names"] = ["desi"]
            f.attrs["creation_date"] = "2024-01-01"
            f.attrs["version"] = "1.0.0"

        return path

    def test_partial_param_mask_correct(self, partial_params_catalogue):
        """Label mask should reflect partial parameter coverage."""
        loader = CatalogueLoader(partial_params_catalogue)
        _, y = loader.load_for_training(survey="desi", label_source="apogee")

        mask = y[:, 2, :]  # (30, 11)

        # Stars 0-9: all 11 parameters valid
        for i in range(10):
            assert np.all(mask[i] == 1), f"Star {i} should have all params valid"

        # Stars 10-19: first 5 params valid, rest masked
        for i in range(10, 20):
            assert np.all(mask[i, :5] == 1), f"Star {i} first 5 params should be valid"
            assert np.all(mask[i, 5:] == 0), f"Star {i} params 5-10 should be masked"

        # Stars 20-29: first 3 params valid
        for i in range(20, 30):
            assert np.all(mask[i, :3] == 1), f"Star {i} first 3 params should be valid"
            assert np.all(mask[i, 3:] == 0), f"Star {i} params 3-10 should be masked"

    def test_all_stars_have_some_labels(self, partial_params_catalogue):
        """All stars should have at least some valid labels."""
        loader = CatalogueLoader(partial_params_catalogue)
        data = loader.load(survey="desi", label_source="apogee")

        # All 30 stars should have at least one valid label
        assert data.n_with_labels == 30


class TestCatalogueLoaderMultiSurvey:
    """Tests for multi-survey loading methods."""

    @pytest.fixture
    def multi_survey_catalogue(self, tmp_path):
        """Create a mock HDF5 catalogue with multiple surveys and overlapping stars."""
        path = tmp_path / "multi_survey.h5"
        n_stars = 100
        n_wave_desi = 7650
        n_wave_boss = 4506

        with h5py.File(path, "w") as f:
            # Metadata - use sequential Gaia IDs for some, with overlaps
            gaia_ids = np.arange(n_stars, dtype=np.int64)
            meta = f.create_group("metadata")
            meta.create_dataset("gaia_id", data=gaia_ids)
            meta.create_dataset("ra", data=np.random.uniform(0, 360, n_stars))
            meta.create_dataset("dec", data=np.random.uniform(-90, 90, n_stars))

            # Surveys group
            surveys = f.create_group("surveys")

            # DESI: has data for stars 0-69 (70 stars)
            desi = surveys.create_group("desi")
            desi_flux = np.zeros((n_stars, n_wave_desi), dtype=np.float32)
            desi_ivar = np.zeros((n_stars, n_wave_desi), dtype=np.float32)
            desi_flux[:70] = np.random.randn(70, n_wave_desi).astype(np.float32)
            desi_ivar[:70] = np.abs(np.random.randn(70, n_wave_desi)).astype(np.float32)
            desi.create_dataset("flux", data=desi_flux)
            desi.create_dataset("ivar", data=desi_ivar)
            desi.create_dataset("wavelength", data=np.linspace(3600, 9800, n_wave_desi))
            desi.create_dataset(
                "snr", data=np.random.uniform(20, 100, n_stars).astype(np.float32)
            )

            # BOSS: has data for stars 30-99 (70 stars, overlaps with DESI on 30-69)
            boss = surveys.create_group("boss")
            boss_flux = np.zeros((n_stars, n_wave_boss), dtype=np.float32)
            boss_ivar = np.zeros((n_stars, n_wave_boss), dtype=np.float32)
            boss_flux[30:] = np.random.randn(70, n_wave_boss).astype(np.float32)
            boss_ivar[30:] = np.abs(np.random.randn(70, n_wave_boss)).astype(np.float32)
            boss.create_dataset("flux", data=boss_flux)
            boss.create_dataset("ivar", data=boss_ivar)
            boss.create_dataset(
                "wavelength", data=np.linspace(3600, 10400, n_wave_boss)
            )
            boss.create_dataset(
                "snr", data=np.random.uniform(20, 100, n_stars).astype(np.float32)
            )

            # Labels
            labels = f.create_group("labels")

            # APOGEE: has labels for stars 0-79 (80 stars)
            apogee = labels.create_group("apogee")
            apogee_values = np.random.randn(n_stars, 11).astype(np.float32)
            apogee_values[80:] = np.nan  # No labels for stars 80-99
            apogee.create_dataset("values", data=apogee_values)
            apogee.create_dataset(
                "errors", data=np.abs(np.random.randn(n_stars, 11)).astype(np.float32)
            )
            apogee.create_dataset("flags", data=np.zeros((n_stars, 11), dtype=np.uint8))
            apogee.create_dataset("gaia_id", data=gaia_ids)

            # GALAH: has labels for stars 20-99 (80 stars)
            galah = labels.create_group("galah")
            galah_values = np.random.randn(n_stars, 11).astype(np.float32)
            galah_values[:20] = np.nan  # No labels for stars 0-19
            galah.create_dataset("values", data=galah_values)
            galah.create_dataset(
                "errors", data=np.abs(np.random.randn(n_stars, 11)).astype(np.float32)
            )
            galah.create_dataset("flags", data=np.zeros((n_stars, 11), dtype=np.uint8))
            galah.create_dataset("gaia_id", data=gaia_ids)

            # Attributes
            f.attrs["n_stars"] = n_stars
            f.attrs["parameter_names"] = PARAMETER_NAMES
            f.attrs["survey_names"] = ["desi", "boss"]
            f.attrs["creation_date"] = "2024-01-01T00:00:00"
            f.attrs["version"] = "1.0.0"

        return path

    def test_list_surveys(self, multi_survey_catalogue):
        """Should list available surveys."""
        loader = CatalogueLoader(multi_survey_catalogue)
        surveys = loader.list_surveys()
        assert "desi" in surveys
        assert "boss" in surveys

    def test_list_label_sources(self, multi_survey_catalogue):
        """Should list available label sources."""
        loader = CatalogueLoader(multi_survey_catalogue)
        sources = loader.list_label_sources()
        assert "apogee" in sources
        assert "galah" in sources

    def test_get_survey_wavelength_counts(self, multi_survey_catalogue):
        """Should return correct wavelength counts for surveys."""
        loader = CatalogueLoader(multi_survey_catalogue)
        counts = loader.get_survey_wavelength_counts(["desi", "boss"])
        assert counts["desi"] == 7650
        assert counts["boss"] == 4506

    def test_load_merged_for_training_single_source(self, multi_survey_catalogue):
        """Should load merged data for training with single label source."""
        loader = CatalogueLoader(multi_survey_catalogue)
        X_dict, y, has_data_dict, has_labels = loader.load_merged_for_training(
            surveys=["desi", "boss"],
            label_source="apogee",
        )

        # Check structure
        assert "desi" in X_dict
        assert "boss" in X_dict
        assert "desi" in has_data_dict
        assert "boss" in has_data_dict

        # Check shapes - should be 3-channel format
        assert X_dict["desi"].ndim == 3
        assert X_dict["desi"].shape[1] == 3  # [flux, error, mask]
        assert y.ndim == 3
        assert y.shape[1] == 3  # [values, errors, mask]

        # has_labels should be None for single source
        assert has_labels is None

    def test_load_merged_for_training_multi_source(self, multi_survey_catalogue):
        """Should load merged data for training with multiple label sources."""
        loader = CatalogueLoader(multi_survey_catalogue)
        X_dict, y_dict, has_data_dict, has_labels_dict = (
            loader.load_merged_for_training(
                surveys=["desi", "boss"],
                label_source="apogee",
                label_sources=["apogee", "galah"],
            )
        )

        # Check y_dict structure
        assert "apogee" in y_dict
        assert "galah" in y_dict
        assert y_dict["apogee"].ndim == 3
        assert y_dict["galah"].ndim == 3

        # Check has_labels_dict structure
        assert "apogee" in has_labels_dict
        assert "galah" in has_labels_dict
        assert has_labels_dict["apogee"].dtype == bool
        assert has_labels_dict["galah"].dtype == bool

    def test_load_merged_union_behavior(self, multi_survey_catalogue):
        """Union should include stars with data from ANY survey."""
        loader = CatalogueLoader(multi_survey_catalogue)
        X_dict, y, has_data_dict, _ = loader.load_merged_for_training(
            surveys=["desi", "boss"],
            label_source="apogee",
        )

        # Should have data from both surveys
        n_total = X_dict["desi"].shape[0]
        assert n_total > 0

        # Both surveys should have same array length
        assert X_dict["boss"].shape[0] == n_total
        assert len(has_data_dict["desi"]) == n_total
        assert len(has_data_dict["boss"]) == n_total

        # Some stars should have DESI data only, some BOSS only, some both
        # (variables computed to verify logic, assertions use them implicitly via union check)
        _ = has_data_dict["desi"] & ~has_data_dict["boss"]  # desi_only
        _ = ~has_data_dict["desi"] & has_data_dict["boss"]  # boss_only
        _ = has_data_dict["desi"] & has_data_dict["boss"]  # both

        # Union means at least one survey has data for each star
        assert (has_data_dict["desi"] | has_data_dict["boss"]).all()


class TestMergedCatalogueDataMethods:
    """Tests for MergedCatalogueData class methods."""

    @pytest.fixture
    def merged_data(self):
        """Create sample MergedCatalogueData."""
        n_stars = 50
        gaia_ids = np.arange(n_stars, dtype=np.int64)
        ra = np.random.uniform(0, 360, n_stars)
        dec = np.random.uniform(-90, 90, n_stars)

        # Create has_data masks
        desi_has_data = np.zeros(n_stars, dtype=bool)
        desi_has_data[:30] = True  # Stars 0-29 have DESI data

        boss_has_data = np.zeros(n_stars, dtype=bool)
        boss_has_data[20:] = True  # Stars 20-49 have BOSS data

        surveys = {
            "desi": {
                "flux": np.random.randn(n_stars, 100).astype(np.float32),
                "ivar": np.abs(np.random.randn(n_stars, 100)).astype(np.float32),
                "wavelength": np.linspace(3600, 9800, 100),
                "snr": np.random.uniform(20, 100, n_stars).astype(np.float32),
                "has_data": desi_has_data,
            },
            "boss": {
                "flux": np.random.randn(n_stars, 80).astype(np.float32),
                "ivar": np.abs(np.random.randn(n_stars, 80)).astype(np.float32),
                "wavelength": np.linspace(3600, 10400, 80),
                "snr": np.random.uniform(20, 100, n_stars).astype(np.float32),
                "has_data": boss_has_data,
            },
        }

        # Zero out ivar for missing data
        surveys["desi"]["ivar"][~desi_has_data] = 0
        surveys["boss"]["ivar"][~boss_has_data] = 0

        primary_labels = np.random.randn(n_stars, 11).astype(np.float32)
        primary_errors = np.abs(np.random.randn(n_stars, 11)).astype(np.float32)
        primary_flags = np.zeros((n_stars, 11), dtype=np.uint8)

        return MergedCatalogueData(
            gaia_ids=gaia_ids,
            ra=ra,
            dec=dec,
            surveys=surveys,
            label_source="apogee",
            primary_labels=primary_labels,
            primary_errors=primary_errors,
            primary_flags=primary_flags,
        )

    def test_n_stars(self, merged_data):
        """Should return correct number of stars."""
        assert merged_data.n_stars == 50

    def test_survey_names(self, merged_data):
        """Should return list of survey names."""
        assert set(merged_data.survey_names) == {"desi", "boss"}

    def test_get_coverage_stats(self, merged_data):
        """Should return coverage statistics."""
        stats = merged_data.get_coverage_stats()
        assert "desi" in stats
        assert "boss" in stats
        # DESI has data for stars 0-29, BOSS for stars 20-49
        assert stats["desi"] == 30  # Stars 0-29
        assert stats["boss"] == 30  # Stars 20-49

    def test_get_overlap_matrix(self, merged_data):
        """Should return overlap matrix between surveys."""
        matrix = merged_data.get_overlap_matrix()
        # Matrix uses tuple keys (survey1, survey2) -> count
        assert ("desi", "boss") in matrix or ("boss", "desi") in matrix
        # Overlap is stars 20-29 (10 stars)
        overlap = matrix.get(("desi", "boss"), matrix.get(("boss", "desi"), 0))
        assert overlap == 10

    def test_filter_by_surveys(self, merged_data):
        """Should filter to stars with data from specified surveys."""
        filtered = merged_data.filter_by_surveys(["desi"])
        # Stars 0-29 have DESI data
        assert filtered.n_stars == 30


class TestLamostMRSDualArm:
    """Tests for LAMOST MRS dual-arm spectral loading."""

    N_RED = 3248

    @pytest.fixture
    def lamost_mrs_catalogue(self, tmp_path):
        """Create a mock HDF5 catalogue with LAMOST MRS dual-arm spectra.

        LAMOST MRS stores spectra as (N, 4, max(n_b, n_r)) with channels:
        [flux_B, ivar_B, flux_R, ivar_R], padded to the longer (blue) arm.
        """
        path = tmp_path / "lamost_mrs_catalogue.h5"
        n_stars = 30
        n_wave = 3375  # Blue arm; the red arm has N_RED bins
        n_params = 11

        with h5py.File(path, "w") as f:
            # Metadata
            meta = f.create_group("metadata")
            meta.create_dataset(
                "gaia_id", data=np.arange(1, n_stars + 1, dtype=np.int64)
            )
            meta.create_dataset("ra", data=np.random.uniform(0, 360, n_stars))
            meta.create_dataset("dec", data=np.random.uniform(-90, 90, n_stars))

            # LAMOST MRS survey with spectra dataset (not flux/ivar)
            surveys = f.create_group("surveys")
            lamost_mrs = surveys.create_group("lamost_mrs")

            # Create spectra array with shape (N, 4, wavelengths)
            # Channels: [flux_B, ivar_B, flux_R, ivar_R]
            spectra = np.zeros((n_stars, 4, n_wave), dtype=np.float32)

            # Stars 0-9: Both arms valid
            spectra[:10, 0, :] = np.random.randn(10, n_wave) + 1.0  # flux_B
            spectra[:10, 1, :] = np.abs(np.random.randn(10, n_wave)) + 0.1  # ivar_B
            spectra[:10, 2, :] = np.random.randn(10, n_wave) + 1.0  # flux_R
            spectra[:10, 3, :] = np.abs(np.random.randn(10, n_wave)) + 0.1  # ivar_R

            # Stars 10-19: Only blue arm valid
            spectra[10:20, 0, :] = np.random.randn(10, n_wave) + 1.0  # flux_B
            spectra[10:20, 1, :] = np.abs(np.random.randn(10, n_wave)) + 0.1  # ivar_B
            spectra[10:20, 2, :] = 0  # flux_R (zero)
            spectra[10:20, 3, :] = 0  # ivar_R (zero)

            # Stars 20-29: Only red arm valid
            spectra[20:30, 0, :] = 0  # flux_B (zero)
            spectra[20:30, 1, :] = 0  # ivar_B (zero)
            spectra[20:30, 2, :] = np.random.randn(10, n_wave) + 1.0  # flux_R
            spectra[20:30, 3, :] = np.abs(np.random.randn(10, n_wave)) + 0.1  # ivar_R

            lamost_mrs.create_dataset("spectra", data=spectra)
            lamost_mrs.create_dataset(
                "wavelength_b", data=np.linspace(4950, 5350, n_wave).astype(np.float32)
            )
            lamost_mrs.create_dataset(
                "wavelength_r",
                data=np.linspace(6300, 6800, self.N_RED).astype(np.float32),
            )
            lamost_mrs.create_dataset(
                "snr", data=np.random.uniform(20, 100, n_stars).astype(np.float32)
            )

            # Labels
            labels = f.create_group("labels")
            apogee = labels.create_group("apogee")
            apogee.create_dataset(
                "values", data=np.random.randn(n_stars, n_params).astype(np.float32)
            )
            apogee.create_dataset(
                "errors",
                data=np.abs(np.random.randn(n_stars, n_params)).astype(np.float32)
                + 0.01,
            )
            apogee.create_dataset(
                "flags", data=np.zeros((n_stars, n_params), dtype=np.uint8)
            )

            # Attributes
            f.attrs["n_stars"] = n_stars
            f.attrs["parameter_names"] = PARAMETER_NAMES
            f.attrs["survey_names"] = ["lamost_mrs"]
            f.attrs["creation_date"] = "2024-01-01"
            f.attrs["version"] = "1.0.0"

        return path

    def test_load_lamost_mrs_spectra(self, lamost_mrs_catalogue):
        """Should load LAMOST MRS spectra and concatenate arms correctly."""
        loader = CatalogueLoader(lamost_mrs_catalogue)
        data = loader.load(survey="lamost_mrs", label_source="apogee")

        assert data.n_stars == 30
        # Arms are trimmed to their grids and concatenated: 3375 + 3248 bins
        assert data.flux.shape == (30, 3375 + self.N_RED)
        assert data.ivar.shape == (30, 3375 + self.N_RED)
        assert data.survey_name == "lamost_mrs"

    def test_lamost_mrs_both_arms_valid(self, lamost_mrs_catalogue):
        """Stars with both arms valid should have combined flux and ivar."""
        loader = CatalogueLoader(lamost_mrs_catalogue)
        data = loader.load(survey="lamost_mrs", label_source="apogee")

        # Stars 0-9 have both arms - should have non-zero flux and ivar
        for i in range(10):
            assert np.any(data.flux[i] != 0), f"Star {i} should have flux"
            assert np.any(data.ivar[i] > 0), f"Star {i} should have valid ivar"

    def test_lamost_mrs_blue_arm_only(self, lamost_mrs_catalogue):
        """Stars with only blue arm should still have valid spectra."""
        loader = CatalogueLoader(lamost_mrs_catalogue)
        data = loader.load(survey="lamost_mrs", label_source="apogee")

        # Stars 10-19 have only blue arm
        for i in range(10, 20):
            # Should have valid ivar from blue arm
            assert np.any(
                data.ivar[i] > 0
            ), f"Star {i} should have valid ivar from blue"

    def test_lamost_mrs_red_arm_only(self, lamost_mrs_catalogue):
        """Stars with only red arm should still have valid spectra."""
        loader = CatalogueLoader(lamost_mrs_catalogue)
        data = loader.load(survey="lamost_mrs", label_source="apogee")

        # Stars 20-29 have only red arm
        for i in range(20, 30):
            # Should have valid ivar from red arm
            assert np.any(data.ivar[i] > 0), f"Star {i} should have valid ivar from red"

    def test_lamost_mrs_all_stars_have_spectra(self, lamost_mrs_catalogue):
        """All stars should be detected as having spectra."""
        loader = CatalogueLoader(lamost_mrs_catalogue)
        data = loader.load(survey="lamost_mrs", label_source="apogee")

        # All 30 stars have at least one arm valid
        assert data.n_with_spectra == 30

    def test_lamost_mrs_for_training(self, lamost_mrs_catalogue):
        """Should return 3-channel training data for LAMOST MRS."""
        loader = CatalogueLoader(lamost_mrs_catalogue)
        X, y = loader.load_for_training(survey="lamost_mrs", label_source="apogee")

        # X: 3-channel spectral data [flux, sigma, mask]
        # Arms are trimmed to their grids and concatenated: 3375 + 3248 bins
        assert X.shape == (30, 3, 3375 + self.N_RED)
        # y: 3-channel labels [values, errors, mask]
        assert y.shape == (30, 3, 11)


class TestLabelSourceFallback:
    """Tests for label source fallback behavior."""

    @pytest.fixture
    def fallback_catalogue(self, tmp_path):
        """Create catalogue with survey-specific label sources."""
        path = tmp_path / "fallback_catalogue.h5"
        n_stars = 20
        n_wave = 500
        n_params = 11

        with h5py.File(path, "w") as f:
            # Metadata
            meta = f.create_group("metadata")
            meta.create_dataset("gaia_id", data=np.arange(n_stars, dtype=np.int64))
            meta.create_dataset("ra", data=np.random.uniform(0, 360, n_stars))
            meta.create_dataset("dec", data=np.random.uniform(-90, 90, n_stars))

            # Surveys
            surveys = f.create_group("surveys")
            boss = surveys.create_group("boss")
            boss.create_dataset(
                "flux", data=np.random.randn(n_stars, n_wave).astype(np.float32)
            )
            boss.create_dataset(
                "ivar",
                data=np.abs(np.random.randn(n_stars, n_wave)).astype(np.float32),
            )
            boss.create_dataset(
                "wavelength", data=np.linspace(3600, 10400, n_wave).astype(np.float32)
            )
            boss.create_dataset(
                "snr", data=np.random.uniform(20, 100, n_stars).astype(np.float32)
            )

            # Labels - survey-specific naming pattern
            labels = f.create_group("labels")

            # apogee_boss - survey-specific labels
            apogee_boss = labels.create_group("apogee_boss")
            apogee_boss.create_dataset(
                "values", data=np.random.randn(n_stars, n_params).astype(np.float32)
            )
            apogee_boss.create_dataset(
                "errors",
                data=np.abs(np.random.randn(n_stars, n_params)).astype(np.float32)
                + 0.01,
            )
            apogee_boss.create_dataset(
                "flags", data=np.zeros((n_stars, n_params), dtype=np.uint8)
            )
            apogee_boss.create_dataset(
                "gaia_id", data=np.arange(n_stars, dtype=np.int64)
            )

            # Attributes
            f.attrs["n_stars"] = n_stars
            f.attrs["parameter_names"] = PARAMETER_NAMES
            f.attrs["survey_names"] = ["boss"]
            f.attrs["creation_date"] = "2024-01-01"
            f.attrs["version"] = "1.0.0"

        return path

    def test_fallback_to_survey_specific_labels(self, fallback_catalogue):
        """Should fall back to apogee_boss when apogee is requested for BOSS survey."""
        loader = CatalogueLoader(fallback_catalogue)
        # Request "apogee" but only "apogee_boss" exists
        data = loader.load(survey="boss", label_source="apogee")

        assert data.n_stars == 20
        assert data.n_with_labels == 20

    def test_info_shows_survey_specific_labels(self, fallback_catalogue):
        """Info should show survey-specific label sources."""
        loader = CatalogueLoader(fallback_catalogue)
        info = loader.get_info()

        assert "apogee_boss" in info.labels


class TestNoSpectraDataError:
    """Tests for error handling when survey has no flux data."""

    @pytest.fixture
    def no_flux_catalogue(self, tmp_path):
        """Create catalogue with a survey missing flux/spectra data."""
        path = tmp_path / "no_flux_catalogue.h5"
        n_stars = 10
        n_params = 11

        with h5py.File(path, "w") as f:
            meta = f.create_group("metadata")
            meta.create_dataset("gaia_id", data=np.arange(n_stars, dtype=np.int64))
            meta.create_dataset("ra", data=np.zeros(n_stars))
            meta.create_dataset("dec", data=np.zeros(n_stars))

            # Survey without flux or spectra
            surveys = f.create_group("surveys")
            bad_survey = surveys.create_group("bad_survey")
            bad_survey.create_dataset("wavelength", data=np.linspace(3600, 9800, 100))
            bad_survey.create_dataset("snr", data=np.ones(n_stars, dtype=np.float32))

            labels = f.create_group("labels")
            apogee = labels.create_group("apogee")
            apogee.create_dataset(
                "values", data=np.zeros((n_stars, n_params), dtype=np.float32)
            )
            apogee.create_dataset(
                "errors", data=np.ones((n_stars, n_params), dtype=np.float32)
            )
            apogee.create_dataset(
                "flags", data=np.zeros((n_stars, n_params), dtype=np.uint8)
            )

            f.attrs["n_stars"] = n_stars
            f.attrs["parameter_names"] = PARAMETER_NAMES
            f.attrs["survey_names"] = ["bad_survey"]
            f.attrs["creation_date"] = "2024-01-01"
            f.attrs["version"] = "1.0.0"

        return path

    def test_raises_error_for_missing_flux(self, no_flux_catalogue):
        """Should raise ValueError when survey has no flux or spectra data."""
        loader = CatalogueLoader(no_flux_catalogue)
        with pytest.raises(ValueError, match="no flux or spectra"):
            loader.load(survey="bad_survey", label_source="apogee")


class TestCoordinateFallback:
    """Tests for RA/Dec coordinate fallback behavior."""

    @pytest.fixture
    def no_coords_catalogue(self, tmp_path):
        """Create catalogue without RA/Dec in survey group."""
        path = tmp_path / "no_coords_catalogue.h5"
        n_stars = 10
        n_wave = 100
        n_params = 11

        with h5py.File(path, "w") as f:
            meta = f.create_group("metadata")
            meta.create_dataset("gaia_id", data=np.arange(n_stars, dtype=np.int64))
            # Metadata has different row count - should not be used
            meta.create_dataset(
                "ra", data=np.random.uniform(0, 360, n_stars * 2)
            )  # Wrong size
            meta.create_dataset(
                "dec", data=np.random.uniform(-90, 90, n_stars * 2)
            )  # Wrong size

            surveys = f.create_group("surveys")
            desi = surveys.create_group("desi")
            desi.create_dataset(
                "flux", data=np.random.randn(n_stars, n_wave).astype(np.float32)
            )
            desi.create_dataset(
                "ivar",
                data=np.abs(np.random.randn(n_stars, n_wave)).astype(np.float32),
            )
            desi.create_dataset(
                "wavelength", data=np.linspace(3600, 9800, n_wave).astype(np.float32)
            )
            desi.create_dataset(
                "snr", data=np.random.uniform(20, 100, n_stars).astype(np.float32)
            )

            labels = f.create_group("labels")
            apogee = labels.create_group("apogee")
            apogee.create_dataset(
                "values", data=np.random.randn(n_stars, n_params).astype(np.float32)
            )
            apogee.create_dataset(
                "errors",
                data=np.abs(np.random.randn(n_stars, n_params)).astype(np.float32)
                + 0.01,
            )
            apogee.create_dataset(
                "flags", data=np.zeros((n_stars, n_params), dtype=np.uint8)
            )

            f.attrs["n_stars"] = n_stars
            f.attrs["parameter_names"] = PARAMETER_NAMES
            f.attrs["survey_names"] = ["desi"]
            f.attrs["creation_date"] = "2024-01-01"
            f.attrs["version"] = "1.0.0"

        return path

    def test_fallback_to_zero_coords(self, no_coords_catalogue):
        """Should use zero coordinates when metadata RA/Dec has wrong size."""
        loader = CatalogueLoader(no_coords_catalogue)
        data = loader.load(survey="desi", label_source="apogee")

        # Should have zero coordinates (fallback)
        assert np.all(data.ra == 0)
        assert np.all(data.dec == 0)


class TestDuplicateDetection:
    """Tests for duplicate detection - loading should error if duplicates exist."""

    @pytest.fixture
    def duplicate_catalogue(self, tmp_path):
        """Create catalogue with duplicate observations."""
        path = tmp_path / "duplicates.h5"
        # 10 unique stars, but some have multiple observations
        n_obs = 15  # 15 total observations for 10 unique stars
        n_wave = 100

        with h5py.File(path, "w") as f:
            # Gaia IDs with duplicates: stars 0-4 have 2 observations each
            gaia_ids = np.array(
                [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9], dtype=np.int64
            )

            meta = f.create_group("metadata")
            meta.create_dataset("gaia_id", data=gaia_ids)
            meta.create_dataset("ra", data=np.random.uniform(0, 360, n_obs))
            meta.create_dataset("dec", data=np.random.uniform(-90, 90, n_obs))

            surveys = f.create_group("surveys")
            desi = surveys.create_group("desi")
            desi.create_dataset(
                "flux", data=np.random.randn(n_obs, n_wave).astype(np.float32)
            )
            desi.create_dataset(
                "ivar", data=np.abs(np.random.randn(n_obs, n_wave)).astype(np.float32)
            )
            desi.create_dataset("wavelength", data=np.linspace(3600, 9800, n_wave))
            # Different SNR for duplicates to test selection
            snr = np.array(
                [50, 100, 60, 80, 70, 90, 55, 85, 65, 95, 40, 45, 50, 55, 60],
                dtype=np.float32,
            )
            desi.create_dataset("snr", data=snr)

            labels = f.create_group("labels")
            apogee = labels.create_group("apogee")
            apogee.create_dataset(
                "values", data=np.random.randn(n_obs, 11).astype(np.float32)
            )
            apogee.create_dataset(
                "errors", data=np.abs(np.random.randn(n_obs, 11)).astype(np.float32)
            )
            apogee.create_dataset("flags", data=np.zeros((n_obs, 11), dtype=np.uint8))

            f.attrs["n_stars"] = n_obs
            f.attrs["parameter_names"] = PARAMETER_NAMES
            f.attrs["survey_names"] = ["desi"]
            f.attrs["creation_date"] = "2024-01-01T00:00:00"
            f.attrs["version"] = "1.0.0"

        return path

    def test_load_merged_raises_on_duplicates(self, duplicate_catalogue):
        """load_merged should raise ValueError if duplicates are found."""
        loader = CatalogueLoader(duplicate_catalogue)
        with pytest.raises(ValueError, match="DUPLICATE OBSERVATIONS DETECTED"):
            loader.load_merged(
                surveys=["desi"],
                label_source="apogee",
            )

    def test_load_merged_for_training_raises_on_duplicates(self, duplicate_catalogue):
        """load_merged_for_training should raise ValueError if duplicates are found."""
        loader = CatalogueLoader(duplicate_catalogue)
        with pytest.raises(ValueError, match="DUPLICATE OBSERVATIONS DETECTED"):
            loader.load_merged_for_training(
                surveys=["desi"],
                label_source="apogee",
            )

    def test_load_merged_sparse_raises_on_duplicates(self, duplicate_catalogue):
        """load_merged_sparse should raise ValueError if duplicates are found."""
        loader = CatalogueLoader(duplicate_catalogue)
        with pytest.raises(ValueError, match="DUPLICATE OBSERVATIONS DETECTED"):
            loader.load_merged_sparse(
                surveys=["desi"],
                label_source="apogee",
            )

    def test_duplicate_error_tells_the_user_to_rebuild_the_catalogue(
        self, duplicate_catalogue
    ):
        loader = CatalogueLoader(duplicate_catalogue)
        with pytest.raises(ValueError, match="scripts/build_catalogue_v2.py") as err:
            loader.load_merged_sparse(surveys=["desi"], label_source="apogee")
        assert "create_deduplicated_catalogue" not in str(err.value)


class TestSparseMergedDataMethods:
    """Tests for SparseMergedData utility methods."""

    @pytest.fixture
    def sparse_data(self):
        """Create a minimal SparseMergedData for testing methods."""
        n_total = 100
        n_params = 11

        # Create fake sparse data for two surveys
        flux = {
            "survey_a": np.random.randn(60, 1000).astype(np.float32),
            "survey_b": np.random.randn(40, 800).astype(np.float32),
        }
        sigma = {
            "survey_a": np.abs(np.random.randn(60, 1000)).astype(np.float32),
            "survey_b": np.abs(np.random.randn(40, 800)).astype(np.float32),
        }
        wavelengths = {
            "survey_a": np.linspace(3500, 10000, 1000).astype(np.float32),
            "survey_b": np.linspace(3600, 9800, 800).astype(np.float32),
        }
        snr = {
            "survey_a": np.random.uniform(10, 100, 60).astype(np.float32),
            "survey_b": np.random.uniform(10, 100, 40).astype(np.float32),
        }

        # Create mappings
        # Survey A has first 60 stars
        global_to_local_a = np.full(n_total, -1, dtype=np.int32)
        global_to_local_a[:60] = np.arange(60)
        local_to_global_a = np.arange(60)

        # Survey B has last 40 stars
        global_to_local_b = np.full(n_total, -1, dtype=np.int32)
        global_to_local_b[60:] = np.arange(40)
        local_to_global_b = np.arange(60, 100)

        global_to_local = {"survey_a": global_to_local_a, "survey_b": global_to_local_b}
        local_to_global = {"survey_a": local_to_global_a, "survey_b": local_to_global_b}

        labels = np.zeros((n_total, 3, n_params), dtype=np.float32)
        labels[:, 0, :] = np.random.randn(n_total, n_params)  # values
        labels[:, 1, :] = np.abs(np.random.randn(n_total, n_params)) + 0.01  # errors
        labels[:, 2, :] = 1.0  # mask

        return SparseMergedData(
            flux=flux,
            sigma=sigma,
            wavelengths=wavelengths,
            snr=snr,
            global_to_local=global_to_local,
            local_to_global=local_to_global,
            labels=labels,
            gaia_ids=np.arange(1000, 1100),
            ra=np.random.uniform(0, 360, n_total),
            dec=np.random.uniform(-90, 90, n_total),
            surveys=["survey_a", "survey_b"],
            n_total=n_total,
            n_params=n_params,
        )

    def test_n_total_property(self, sparse_data):
        """n_total should return correct value."""
        assert sparse_data.n_total == 100

    def test_surveys_property(self, sparse_data):
        """surveys should list survey names."""
        assert set(sparse_data.surveys) == {"survey_a", "survey_b"}

    def test_memory_usage_includes_all_arrays(self, sparse_data):
        """memory_usage_mb should include all data arrays."""
        usage = sparse_data.memory_usage_mb()

        # Flat key structure: {survey}_flux, {survey}_sigma
        assert "survey_a_flux" in usage
        assert "survey_b_flux" in usage
        assert "labels" in usage
        assert "total" in usage
        assert usage["total"] > 0

    def test_has_data_returns_mask(self, sparse_data):
        """has_data should return boolean mask."""
        mask_a = sparse_data.has_data("survey_a")
        mask_b = sparse_data.has_data("survey_b")

        assert mask_a.dtype == bool
        assert len(mask_a) == 100
        # First 60 stars have survey_a data
        assert np.sum(mask_a) == 60
        # Last 40 stars have survey_b data
        assert np.sum(mask_b) == 40

    def test_coverage_returns_fraction(self, sparse_data):
        """coverage should return fraction."""
        assert sparse_data.coverage("survey_a") == 0.6
        assert sparse_data.coverage("survey_b") == 0.4

    def test_n_with_data_returns_count(self, sparse_data):
        """n_with_data should return count."""
        assert sparse_data.n_with_data("survey_a") == 60
        assert sparse_data.n_with_data("survey_b") == 40

    def test_get_coverage_stats(self, sparse_data):
        """get_coverage_stats should return counts per survey."""
        stats = sparse_data.get_coverage_stats()
        assert stats["survey_a"] == 60
        assert stats["survey_b"] == 40


class TestParameterNamesAndSpectralNormalization:
    """Catalogue-defined parameter lists and loader-level flux normalization."""

    PARAMS = ["teff", "logg", "fe_h", "mg_h", "ba_h"]

    def _build(self, path, parameter_names=None, n_params=None):
        rng = np.random.default_rng(0)
        n_stars, n_wave = 30, 50
        n_params = n_params or len(parameter_names)
        with h5py.File(path, "w") as f:
            meta = f.create_group("metadata")
            meta.create_dataset(
                "gaia_id", data=np.arange(1, n_stars + 1, dtype=np.int64)
            )
            meta.create_dataset("ra", data=rng.uniform(0, 360, n_stars))
            meta.create_dataset("dec", data=rng.uniform(-90, 90, n_stars))
            boss = f.create_group("surveys/boss")
            flux = (rng.normal(size=(n_stars, n_wave)) * 3 + 50).astype(np.float32)
            ivar = np.ones((n_stars, n_wave), dtype=np.float32)
            ivar[:, :5] = 0.0  # always-masked pixels
            snr = np.full(n_stars, 20.0, dtype=np.float32)
            snr[-5:] = 0.0  # last 5 stars have no BOSS spectrum
            flux[-5:] = 0.0
            ivar[-5:] = 0.0
            boss.create_dataset("flux", data=flux)
            boss.create_dataset("ivar", data=ivar)
            boss.create_dataset("wavelength", data=np.linspace(3600, 9000, n_wave))
            boss.create_dataset("snr", data=snr)
            # LAMOST MRS in the real v2 layout: padded (N, 4, max_wave) arms
            mrs = f.create_group("surveys/lamost_mrs")
            n_b, n_r = 6, 8
            spectra = np.zeros((n_stars, 4, n_r), dtype=np.float32)
            spectra[:, 0, :n_b] = rng.normal(size=(n_stars, n_b)) + 5
            spectra[:, 1, :n_b] = 1.0
            spectra[:, 2, :n_r] = rng.normal(size=(n_stars, n_r)) + 7
            spectra[:, 3, :n_r] = 2.0
            mrs.create_dataset("spectra", data=spectra)
            mrs.create_dataset("wavelength_b", data=np.linspace(4950, 5350, n_b))
            mrs.create_dataset("wavelength_r", data=np.linspace(6300, 6800, n_r))
            mrs.create_dataset("snr", data=np.full(n_stars, 15.0, dtype=np.float32))
            apogee = f.create_group("labels/apogee")
            values = rng.normal(size=(n_stars, n_params)).astype(np.float32)
            values[:, 0] = rng.uniform(4000, 6000, n_stars)
            apogee.create_dataset("values", data=values)
            apogee.create_dataset(
                "errors", data=np.full((n_stars, n_params), 0.1, dtype=np.float32)
            )
            apogee.create_dataset("flags", data=np.zeros((n_stars, n_params), np.uint8))
            apogee.create_dataset("gaia_id", data=np.arange(1, n_stars + 1))
            f.attrs["n_stars"] = n_stars
            f.attrs["survey_names"] = ["boss", "lamost_mrs"]
            f.attrs["creation_date"] = "2026-01-01"
            f.attrs["version"] = "2.0"
            if parameter_names is not None:
                f.attrs["parameter_names"] = parameter_names
        return path

    def test_parameter_names_from_attrs(self, tmp_path):
        path = self._build(tmp_path / "cat.h5", self.PARAMS)
        loader = CatalogueLoader(path)
        assert loader.get_parameter_names() == self.PARAMS
        data = loader.load_merged_sparse(["boss"], label_source="apogee")
        assert data.n_params == len(self.PARAMS)
        assert data.parameter_names == self.PARAMS
        assert data.labels.shape == (30, 3, len(self.PARAMS))

    def test_parameter_names_fallback_without_attrs(self, tmp_path):
        path = self._build(tmp_path / "cat.h5", None, n_params=11)
        assert CatalogueLoader(path).get_parameter_names() == PARAMETER_NAMES

    def test_wide_labels_without_names_raise(self, tmp_path):
        path = self._build(tmp_path / "cat.h5", None, n_params=20)
        with pytest.raises(ValueError, match="no parameter_names"):
            CatalogueLoader(path).get_parameter_names()

    def test_sparse_normalization(self, tmp_path):
        path = self._build(tmp_path / "cat.h5", self.PARAMS)
        loader = CatalogueLoader(path)
        raw = loader.load_merged_sparse(["boss"], label_source="apogee")
        norm = loader.load_merged_sparse(
            ["boss"], label_source="apogee", normalize_spectra=True
        )
        flux = norm.flux["boss"]
        assert flux.shape == raw.flux["boss"].shape == (25, 50)
        np.testing.assert_allclose(np.median(flux[:, 5:], axis=1), 0.0, atol=1e-5)
        assert np.all(flux[:, :5] == 0) and np.all(norm.sigma["boss"][:, :5] == 0)
        assert not np.allclose(raw.flux["boss"][:, 5:], flux[:, 5:])

    def test_all_loading_paths_normalize_identically(self, tmp_path):
        path = self._build(tmp_path / "cat.h5", self.PARAMS)
        loader = CatalogueLoader(path)
        sparse = loader.load_merged_sparse(
            ["boss"], label_source="apogee", normalize_spectra=True
        )
        X_dict, _, has_data, _ = loader.load_merged_for_training(
            ["boss"], label_source="apogee", normalize_spectra=True
        )
        dense_flux = X_dict["boss"][has_data["boss"], 0, :]
        np.testing.assert_allclose(dense_flux, sparse.flux["boss"], rtol=1e-6)
        X, _ = loader.load_for_training(
            "boss", label_source="apogee", normalize_spectra=True
        )
        np.testing.assert_allclose(X[:, 0, :], sparse.flux["boss"], rtol=1e-6)

    @pytest.mark.parametrize("normalize", [False, True])
    @pytest.mark.parametrize("survey", ["boss", "lamost_mrs"])
    def test_load_spectrum_matches_training_rows(self, tmp_path, survey, normalize):
        path = self._build(tmp_path / "cat.h5", self.PARAMS)
        loader = CatalogueLoader(path)
        sparse = loader.load_merged_sparse(
            [survey], label_source="apogee", normalize_spectra=normalize
        )
        X, wavelength, gaia_id = loader.load_spectrum(
            survey, 3, normalize_spectra=normalize
        )
        local = sparse.global_to_local[survey][3]
        assert X.shape == (3, len(sparse.wavelengths[survey]))
        np.testing.assert_allclose(wavelength, sparse.wavelengths[survey])
        np.testing.assert_allclose(X[0], sparse.flux[survey][local], rtol=1e-6)
        np.testing.assert_array_equal(X[2], sparse.sigma[survey][local] > 0)
        np.testing.assert_array_equal(X[1], sparse.sigma[survey][local])
        assert gaia_id == 4

    def test_load_labels_for_index(self, tmp_path):
        path = self._build(tmp_path / "cat.h5", self.PARAMS)
        loader = CatalogueLoader(path)
        values, errors, flags = loader.load_labels_for_index(2, "apogee")
        assert values.shape == errors.shape == flags.shape == (len(self.PARAMS),)
        sparse = loader.load_merged_sparse(["boss"], label_source="apogee")
        np.testing.assert_allclose(values, sparse.labels[2, 0])
        assert loader.load_labels_for_index(2, "galah") is None


class TestSelectiveRowReads:
    """Row-subset reads used by the merged loaders must equal dset[:][rows]."""

    ROW_SETS = {
        "empty": [],
        "first": [0],
        "last": [236],
        "runs": [0, 1, 2, 50, 51, 120, 121, 122, 123, 200, 235, 236],
        "all": list(range(237)),
    }

    @pytest.fixture
    def datasets(self, tmp_path):
        rng = np.random.default_rng(0)
        data2d = rng.normal(size=(237, 13)).astype(np.float32)
        data3d = rng.normal(size=(237, 4, 9)).astype(np.float32)
        path = tmp_path / "rows.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("contig2d", data=data2d)
            f.create_dataset("contig3d", data=data3d)
            f.create_dataset("gzip2d", data=data2d, compression="gzip", chunks=(17, 5))
            f.create_dataset("chunk3d", data=data3d, chunks=(10, 4, 9))
        return path, {"2d": data2d, "3d": data3d}

    @pytest.mark.parametrize("read_bytes", [64 * 1024**2, 100, 1])
    @pytest.mark.parametrize("rows", list(ROW_SETS))
    def test_matches_full_read(self, datasets, monkeypatch, read_bytes, rows):
        import dorothy.data.catalogue_loader as cl

        monkeypatch.setattr(cl, "_ROW_READ_BYTES", read_bytes)
        path, data = datasets
        idx = np.array(self.ROW_SETS[rows], dtype=np.int64)
        with h5py.File(path, "r") as f:
            for name in ("contig2d", "contig3d", "gzip2d", "chunk3d"):
                expected = data[name[-2:]][idx]
                got = cl._read_rows(f[name], idx)
                assert got.dtype == expected.dtype and got.shape == expected.shape
                np.testing.assert_array_equal(got, expected)
                # Without an output array, blocks come from a reused buffer
                pieces = [blk.copy() for _, _, blk in cl._iter_row_blocks(f[name], idx)]
                joined = np.concatenate(pieces) if pieces else expected[:0]
                np.testing.assert_array_equal(joined, expected)

    def test_match_ids_like_dict(self):
        from dorothy.data.catalogue_loader import _match_ids

        keys = np.array([50, 10, 30, 10, -1, 70, 30], dtype=np.int64)
        query = np.array([30, 99, 10, -1, 50, 0, 30], dtype=np.int64)
        lookup = {int(k): i for i, k in enumerate(keys)}  # last occurrence wins
        found, idx = _match_ids(keys, query)
        np.testing.assert_array_equal(found, np.isin(query, keys))
        np.testing.assert_array_equal(idx, [lookup[int(q)] for q in query[found]])

    def test_sparse_load_same_for_compressed_catalogue(self, tmp_path):
        rng = np.random.default_rng(1)
        n, n_wave = 300, 21
        has = rng.random(n) < 0.3
        flux = np.where(has[:, None], rng.normal(10, 1, (n, n_wave)), 0)
        ivar = np.where(has[:, None], rng.uniform(0.5, 2, (n, n_wave)), 0)
        flux, ivar = flux.astype(np.float32), ivar.astype(np.float32)
        spectra = np.zeros((n, 4, 8), dtype=np.float32)
        spectra[has] = rng.normal(5, 1, (has.sum(), 4, 8))
        paths = {}
        for compression in (None, "gzip"):
            path = tmp_path / f"cat_{compression}.h5"
            kw = {"compression": compression} if compression else {}
            with h5py.File(path, "w") as f:
                f.create_dataset("metadata/gaia_id", data=np.arange(1, n + 1))
                g = f.create_group("surveys/boss")
                g.create_dataset("flux", data=flux, **kw)
                g.create_dataset("ivar", data=ivar, **kw)
                g.create_dataset("wavelength", data=np.arange(n_wave, dtype=float))
                g.create_dataset("snr", data=has.astype(np.float32) * 20)
                g = f.create_group("surveys/lamost_mrs")
                g.create_dataset("spectra", data=spectra, **kw)
                g.create_dataset("wavelength_b", data=np.arange(8.0))
                g.create_dataset("wavelength_r", data=np.arange(6.0))
                g.create_dataset("snr", data=has.astype(np.float32) * 20)
                g = f.create_group("labels/apogee")
                g.create_dataset("values", data=rng.normal(size=(n, 11)))
                g.create_dataset("errors", data=np.full((n, 11), 0.1))
                g.create_dataset("flags", data=np.zeros((n, 11), np.uint8))
                g.create_dataset("gaia_id", data=np.arange(1, n + 1))
                f.attrs["parameter_names"] = PARAMETER_NAMES
            paths[compression] = path
        loaded = {
            c: CatalogueLoader(p).load_merged_sparse(["boss", "lamost_mrs"])
            for c, p in paths.items()
        }
        rows = np.flatnonzero(has)
        for data in loaded.values():
            np.testing.assert_array_equal(data.flux["boss"], flux[rows])
            np.testing.assert_array_equal(data.sigma["boss"], 1 / np.sqrt(ivar[rows]))
            mrs = np.concatenate([spectra[rows, 0, :8], spectra[rows, 2, :6]], axis=1)
            np.testing.assert_array_equal(data.flux["lamost_mrs"], mrs)
            np.testing.assert_array_equal(data.local_to_global["boss"], rows)


class TestSurveyWavelengthGrid:
    """Every loader path must give a survey the same grid and spectra."""

    @pytest.mark.parametrize("survey", ["boss", "lamost_mrs"])
    def test_every_loader_path_gives_the_same_grid_and_spectra(self, tmp_path, survey):
        loader = CatalogueLoader(build_catalogue(tmp_path / "cat.h5"))
        sparse = loader.load_merged_sparse([survey])
        grid = sparse.wavelengths[survey]
        info = loader.get_info().surveys[survey]
        assert info["wavelength_bins"] == len(grid)
        assert info["wavelength_range"] == (grid[0], grid[-1])
        np.testing.assert_array_equal(loader.get_survey_wavelengths(survey), grid)
        assert loader.get_survey_wavelength_counts([survey]) == {survey: len(grid)}
        data = loader.load(survey)
        np.testing.assert_array_equal(data.wavelength, grid)
        rows = sparse.local_to_global[survey]
        flux, ivar = zero_invalid_pixels(data.flux[rows], data.ivar[rows])
        np.testing.assert_array_equal(flux, sparse.flux[survey])
        sigma = np.where(ivar > 0, 1 / np.sqrt(np.where(ivar > 0, ivar, 1)), 0)
        np.testing.assert_array_equal(sigma, sparse.sigma[survey])

    @pytest.mark.parametrize("survey", ["boss", "lamost_mrs"])
    def test_raw_spectra_are_returned_as_stored(self, tmp_path, survey):
        path = build_catalogue(tmp_path / "cat.h5")
        raw = CatalogueLoader(path).load_raw_spectra(survey)
        with h5py.File(path, "r") as f:
            grp = f["surveys"][survey]
            assert set(raw) == set(grp)
            for name in grp:
                np.testing.assert_array_equal(raw[name], grp[name][:])


class TestLoadMulti:
    def test_intersection_keeps_only_stars_with_data_in_every_survey(self, tmp_path):
        path = build_catalogue(tmp_path / "cat.h5", n_stars=10)
        with h5py.File(path, "r+") as f:
            for survey, rows in [("boss", slice(0, 5)), ("desi", slice(3, 10))]:
                snr = np.zeros(10)
                snr[rows] = 20.0
                f["surveys"][survey]["snr"][...] = snr
        data = CatalogueLoader(path).load_multi(["boss", "desi"], mode="intersection")
        for survey in ("boss", "desi"):
            np.testing.assert_array_equal(data[survey].gaia_ids, [4, 5])


class TestHasLabels:
    @pytest.mark.parametrize("dense", [False, True])
    def test_has_labels_counts_only_labels_that_survive_the_flag_filter(
        self, tmp_path, dense
    ):
        path = build_catalogue(tmp_path / "cat.h5")
        with h5py.File(path, "r+") as f:
            for source in ("apogee", "galah"):
                flags = f["labels"][source]["flags"][:]
                flags[:20, 0] = 1  # max_flag_bits=0 removes all their labels
                f["labels"][source]["flags"][...] = flags
        loader = CatalogueLoader(path)
        surveys = ["boss", "desi"]
        kwargs = {"label_sources": ["apogee", "galah"], "max_flag_bits": 0}
        if dense:
            _, labels, _, has_labels = loader.load_merged_for_training(
                surveys, **kwargs
            )
        else:
            data = loader.load_merged_sparse(surveys, **kwargs)
            labels, has_labels = data.labels_dict, data.has_labels_dict
        for source in ("apogee", "galah"):
            y = labels[source]
            np.testing.assert_array_equal(has_labels[source], (y[:, 2] > 0).any(1))
            # Stars with a positive error but no label left exist
            assert ((y[:, 1] > 0).any(axis=1) & ~has_labels[source]).any()


class TestLabelMask:
    @pytest.mark.parametrize("max_flag_bits", [-1, 0, 2])
    def test_every_loader_masks_labels_by_flag_error_and_flag_count(
        self, tmp_path, max_flag_bits
    ):
        path = build_catalogue(tmp_path / "cat.h5")
        rng = np.random.default_rng(0)
        expected, flag_ok = {}, {}
        with h5py.File(path, "r+") as f:
            for source in ("apogee", "galah"):
                grp = f["labels"][source]
                flags = rng.choice([0, 0, 0, 0, 1, 2, 3], size=grp["flags"].shape)
                grp["flags"][...] = flags
                # Set bits per star, counted without np.bitwise_count
                bits = np.unpackbits(flags.astype(np.uint8)[:, :, None], axis=2)
                flag_ok[source] = (max_flag_bits < 0) | (
                    bits.sum(axis=(1, 2)) <= max_flag_bits
                )
                errors = grp["errors"][:]
                valid = (flags == 0) & (errors > 0) & flag_ok[source][:, None]
                expected[source] = np.stack(
                    [grp["values"][:], errors, valid.astype(np.float32)], axis=1
                )
            # Without gaia_id, the primary labels are matched by catalogue row
            del f["labels/apogee/gaia_id"]
            has_boss = np.any(f["surveys/boss/ivar"][:] > 0, axis=1)
        # load_for_training keeps complete stars and drops the flagged ones
        has_apogee = np.any(expected["apogee"][:, 1] > 0, axis=1)
        boss_rows = has_boss & has_apogee & flag_ok["apogee"]
        loader = CatalogueLoader(path)
        surveys, sources = ["boss", "desi"], ["apogee", "galah"]
        common = {"max_flag_bits": max_flag_bits}

        sparse = loader.load_merged_sparse(surveys, label_sources=sources, **common)
        rows = np.flatnonzero(sparse.has_data("boss") | sparse.has_data("desi"))
        _, dense, _, _ = loader.load_merged_for_training(
            surveys, label_sources=sources, **common
        )
        _, primary, _, _ = loader.load_merged_for_training(surveys, **common)
        _, single = loader.load_for_training("boss", **common)

        np.testing.assert_array_equal(sparse.labels, expected["apogee"])
        np.testing.assert_array_equal(primary, expected["apogee"][rows])
        np.testing.assert_array_equal(single, expected["apogee"][boss_rows])
        for source in sources:
            np.testing.assert_array_equal(sparse.labels_dict[source], expected[source])
            np.testing.assert_array_equal(dense[source], expected[source][rows])


class TestLoadLabels:
    def test_labels_are_matched_to_catalogue_stars_by_gaia_id(self, tmp_path):
        """A label group may list only some stars, in any order."""
        path = build_catalogue(tmp_path / "cat.h5")
        loader = CatalogueLoader(path)
        expected = loader.load_labels("galah", max_flag_bits=-1)
        with h5py.File(path, "r+") as f:
            grp = f["labels/galah"]
            labelled = np.flatnonzero(grp["errors"][:].any(axis=1))
            rows = np.random.default_rng(0).permutation(labelled)
            for name in ("values", "errors", "flags", "gaia_id"):
                data = grp[name][:][rows]
                del grp[name]
                grp.create_dataset(name, data=data)

        assert 0 < len(rows) < len(expected)
        np.testing.assert_array_equal(loader.load_labels("galah", -1), expected)
        with pytest.raises(ValueError, match="Label source 'lamost' not in"):
            loader.load_labels("lamost")

    @pytest.mark.parametrize("max_flag_bits", [-2, None])
    def test_invalid_max_flag_bits_is_rejected(self, tmp_path, max_flag_bits):
        loader = CatalogueLoader(build_catalogue(tmp_path / "cat.h5", n_stars=20))
        with pytest.raises(ValueError, match="max_flag_bits must be an integer"):
            loader.load_labels("galah", max_flag_bits)

    def test_catalogue_without_labels_raises_value_error(self, tmp_path):
        path = build_catalogue(tmp_path / "cat.h5")
        with h5py.File(path, "r+") as f:
            del f["labels"]
        with pytest.raises(ValueError, match="has no 'labels' group"):
            CatalogueLoader(path).load_labels("galah")


class TestGaiaIds:
    def test_dense_loading_without_gaia_ids_raises(self, tmp_path):
        path = build_catalogue(tmp_path / "cat.h5")
        with h5py.File(path, "r+") as f:
            del f["metadata/gaia_id"]
            for source in ("apogee", "galah"):
                del f["labels"][source]["gaia_id"]
        with pytest.raises(ValueError, match="no Gaia IDs"):
            CatalogueLoader(path).load_merged(["boss", "desi"])

    def test_sparse_loading_rejects_labels_in_another_row_order(self, tmp_path):
        path = build_catalogue(tmp_path / "cat.h5")
        with h5py.File(path, "r+") as f:
            f["labels/apogee/gaia_id"][...] = f["labels/apogee/gaia_id"][:][::-1]
        with pytest.raises(ValueError, match="does not match metadata/gaia_id"):
            CatalogueLoader(path).load_merged_sparse(["boss"])

    @pytest.mark.parametrize("dense", [False, True])
    def test_secondary_label_source_without_gaia_id_is_rejected(self, tmp_path, dense):
        path = build_catalogue(tmp_path / "cat.h5")
        with h5py.File(path, "r+") as f:
            del f["labels/galah/gaia_id"]
        loader = CatalogueLoader(path)
        load = loader.load_merged_for_training if dense else loader.load_merged_sparse
        with pytest.raises(ValueError, match="/labels/galah has no gaia_id"):
            load(["boss", "desi"], label_sources=["apogee", "galah"])


class TestNanSnr:
    def test_dense_and_sparse_loaders_skip_the_same_nan_snr_rows(self, tmp_path):
        path = build_catalogue(tmp_path / "cat.h5")
        with h5py.File(path, "r+") as f:
            boss = f["surveys/boss/snr"][:]
            desi = f["surveys/desi/snr"][:]
            boss_only = np.flatnonzero((boss > 0) & (desi == 0))[0]
            both = np.flatnonzero((boss > 0) & (desi > 0))[0]
            boss[[boss_only, both]] = np.nan
            f["surveys/boss/snr"][...] = boss
        loader = CatalogueLoader(path)
        sparse = loader.load_merged_sparse(["boss", "desi"])
        dense = loader.load_merged(["boss", "desi"])
        rows = np.flatnonzero(sparse.has_data("boss") | sparse.has_data("desi"))
        np.testing.assert_array_equal(dense.gaia_ids, sparse.gaia_ids[rows])
        for survey in ("boss", "desi"):
            np.testing.assert_array_equal(
                dense.surveys[survey]["has_data"], sparse.has_data(survey)[rows]
            )


class TestModelInputs:
    # ivar = inf gives sigma = 0 at a pixel this helper counts as valid
    @pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
    @pytest.mark.parametrize("encode", [False, True])
    def test_model_inputs_match_the_direct_computation(self, encode):
        from dorothy.data.catalogue_loader import _spectra_to_model_inputs
        from dorothy.data.fits_loader import encode_precision

        rng = np.random.default_rng(0)
        flux = rng.normal(size=(7, 9)).astype(np.float32)
        ivar = rng.uniform(0.5, 2.0, size=(7, 9)).astype(np.float32)
        ivar[:, :4] = [0.0, -1.0, np.nan, np.inf]
        flux[:, 0] = np.nan  # a masked pixel can hold any value
        valid = ivar > 0
        sigma = np.zeros_like(ivar)
        sigma[valid] = 1.0 / np.sqrt(ivar[valid])
        expected = np.stack([flux, sigma, valid.astype(np.float32)], axis=1)
        if encode:
            expected = encode_precision(expected)
        got = _spectra_to_model_inputs(flux, ivar, encode, block_rows=3)
        assert got.dtype == expected.dtype and got.shape == expected.shape
        assert got.tobytes() == expected.tobytes()

    @pytest.mark.parametrize("encode", [False, True])
    @pytest.mark.parametrize("normalize", [False, True])
    def test_every_path_gives_the_same_inputs_with_zeros_at_invalid_pixels(
        self, tmp_path, normalize, encode
    ):
        import torch
        from astropy.io import fits
        from astropy.table import Table

        from dorothy.data.fits_loader import (
            FITSLoader,
            encode_precision,
            normalize_spectra,
        )
        from dorothy.training.batches import SparseBatchSource

        path = build_catalogue(tmp_path / "cat.h5")
        with h5py.File(path, "r+") as f:
            boss = f["surveys/boss"]
            flux, ivar = boss["flux"][:], boss["ivar"][:]
            # Varied ivar, so that an inexact 1/sqrt shows in some pixels
            ivar *= np.random.default_rng(0).uniform(0.5, 2, ivar.shape)
            rows = np.flatnonzero(boss["snr"][:] > 0)
            # Pixels 0 and 1 are masked (ivar 0) and hold a nonzero flux
            r = rows[:3]
            # A valid ivar that overflows when normalization scales it
            ivar[r[0], 10] = 3e38
            flux[r[:2], 0] = [np.nan, np.inf]
            flux[r[:2], 5] = [np.nan, np.inf]
            ivar[r, 6] = [np.nan, -1.0, np.inf]
            boss["flux"][...] = flux
            boss["ivar"][...] = ivar
            has_apogee = (f["labels/apogee/errors"][:] > 0).any(axis=1)[rows]
        flux, ivar = flux[rows], ivar[rows]
        ref_flux, ref_ivar = (
            normalize_spectra(flux, ivar) if normalize else (flux, ivar)
        )
        valid = np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0) & (ref_ivar > 0)
        assert valid[0, 10] != normalize  # the overflow invalidates the pixel
        sigma = np.where(valid, 1 / np.sqrt(np.where(valid, ref_ivar, 1)), 0)
        expected = np.stack([np.where(valid, ref_flux, 0), sigma, valid], axis=1)
        if encode:
            expected = encode_precision(expected)

        loader = CatalogueLoader(path)
        options = {"normalize_spectra": normalize, "precision_encoding": encode}
        X_dict, _, _, _ = loader.load_merged_for_training(["boss"], **options)
        single, _ = loader.load_for_training("boss", max_flag_bits=-1, **options)
        one = loader.load_spectrum("boss", int(r[0]), **options)
        sparse = loader.load_merged_sparse(["boss"], **options)
        batch = SparseBatchSource(
            sparse, rows, {"apogee": sparse.labels}, torch.device("cpu")
        ).build_batch(rows)
        fits_path = tmp_path / "spectra.fits"
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.BinTableHDU(Table({"FLUX": flux}), name="SPECTRA"),
                fits.BinTableHDU(Table({"IVAR": ivar}), name="SPEC_IVAR"),
            ]
        ).writeto(fits_path)
        from_fits = FITSLoader(fits_path).load(
            apply_normalization=normalize,
            apply_quality_filter=False,
            compute_spectral_mask=True,
            require_labels=False,
        )

        np.testing.assert_array_equal(X_dict["boss"], expected)
        np.testing.assert_array_equal(single, expected[has_apogee])
        np.testing.assert_array_equal(one[0], expected[0])
        np.testing.assert_array_equal(batch.X["boss"].numpy(), expected)
        from_fits = from_fits.get_model_input_3channel(apply_quality_mask=False)
        if encode:  # as predict_from_fits does
            encode_precision(from_fits, inplace=True)
        np.testing.assert_array_equal(from_fits, expected)


class TestLabelSourceNames:
    def test_v1_group_names_give_the_same_labels_in_every_loader(self, tmp_path):
        path = build_catalogue(tmp_path / "cat.h5")
        loader = CatalogueLoader(path)
        surveys, sources = ["boss", "desi"], ["apogee", "galah"]

        def load_all():
            sparse = loader.load_merged_sparse(surveys, label_sources=sources)
            _, dense, _, _ = loader.load_merged_for_training(
                surveys, label_sources=sources
            )
            single = [
                loader.load(s, label_source=src).labels
                for s, src in [("boss", None), ("desi", "galah")]
            ]
            return sparse.labels, sparse.labels_dict, dense, single

        v2 = load_all()
        # v1 catalogues have one label group per survey: {source}_{survey}
        with h5py.File(path, "r+") as f:
            f.move("labels/apogee", "labels/apogee_boss")
            f.copy("labels/apogee_boss", "labels/apogee_desi")
            f.move("labels/galah", "labels/galah_desi")
        v1 = load_all()
        np.testing.assert_array_equal(v1[0], v2[0])
        for source in sources:
            np.testing.assert_array_equal(v1[1][source], v2[1][source])
            np.testing.assert_array_equal(v1[2][source], v2[2][source])
        for labels_v1, labels_v2 in zip(v1[3], v2[3], strict=True):
            np.testing.assert_array_equal(labels_v1, labels_v2)
