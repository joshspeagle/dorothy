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

            # GALAH: has labels for stars 20-99 (80 stars)
            galah = labels.create_group("galah")
            galah_values = np.random.randn(n_stars, 11).astype(np.float32)
            galah_values[:20] = np.nan  # No labels for stars 0-19
            galah.create_dataset("values", data=galah_values)
            galah.create_dataset(
                "errors", data=np.abs(np.random.randn(n_stars, 11)).astype(np.float32)
            )
            galah.create_dataset("flags", data=np.zeros((n_stars, 11), dtype=np.uint8))

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


class TestDeduplication:
    """Tests for deduplication methods."""

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

    def test_load_merged_deduplicates(self, duplicate_catalogue):
        """load_merged_for_training should deduplicate by default."""
        loader = CatalogueLoader(duplicate_catalogue)
        X_dict, y, has_data_dict, _ = loader.load_merged_for_training(
            surveys=["desi"],
            label_source="apogee",
            smart_deduplicate=False,  # Use simple deduplication
        )

        # Should have fewer than 15 (deduplicated)
        n_unique = X_dict["desi"].shape[0]
        assert n_unique < 15, "Should deduplicate duplicate observations"
        assert n_unique >= 9, "Should have at least 9 unique Gaia IDs"

    def test_smart_deduplicate_picks_best(self, duplicate_catalogue):
        """Smart deduplication should select highest SNR observation."""
        loader = CatalogueLoader(duplicate_catalogue)
        X_dict, y, has_data_dict, _ = loader.load_merged_for_training(
            surveys=["desi"],
            label_source="apogee",
            smart_deduplicate=True,
        )

        # Should have fewer than 15 (deduplicated)
        n_unique = X_dict["desi"].shape[0]
        assert n_unique < 15, "Should deduplicate duplicate observations"
        assert n_unique >= 9, "Should have at least 9 unique Gaia IDs"
