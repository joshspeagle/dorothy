"""
Tests for the FITS loading infrastructure.

These tests verify:
1. Spectrum normalization (median/IQR scaling)
2. Quality filtering logic
3. SpectralData container class
4. FITSLoader with mock FITS files
5. Data splitting functionality
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from dorothy.data.fits_loader import (
    APOGEE_COLUMN_MAP,
    PARAMETER_BOUNDS,
    FITSLoader,
    SpectralData,
    apply_quality_filter,
    encode_precision,
    ivar_to_error_and_mask,
    normalize_spectra,
    normalize_spectrum,
    split_data,
)


class TestNormalizeSpectra:
    """Tests for batched per-spectrum normalization."""

    def test_rows_normalized_independently(self):
        rng = np.random.default_rng(0)
        flux = rng.normal(size=(5, 200)) * np.arange(1, 6)[:, None] + 10.0
        ivar = np.ones_like(flux)
        norm, scaled = normalize_spectra(flux, ivar)
        assert norm.dtype == np.float32 and scaled.dtype == np.float32
        np.testing.assert_allclose(np.median(norm, axis=1), 0.0, atol=1e-5)
        q25, q75 = np.percentile(norm, [25, 75], axis=1)
        np.testing.assert_allclose(q75 - q25, 1.0, rtol=1e-4)

    def test_invalid_pixels_excluded_and_zeroed(self):
        rng = np.random.default_rng(1)
        flux = rng.normal(size=(1, 100)) + 5.0
        ivar = np.ones_like(flux)
        flux_bad = flux.copy()
        ivar_bad = ivar.copy()
        flux_bad[0, :10] = np.nan  # non-finite flux
        flux_bad[0, 10:20] = 1e6  # huge flux but ivar = 0
        ivar_bad[0, 10:20] = 0.0
        norm, scaled = normalize_spectra(flux_bad, ivar_bad)
        assert np.all(np.isfinite(norm))
        assert np.all(norm[0, :20] == 0) and np.all(scaled[0, :20] == 0)
        # Statistics come from the 80 valid pixels only
        ref, _ = normalize_spectra(flux[:, 20:], ivar[:, 20:])
        np.testing.assert_allclose(norm[0, 20:], ref[0], rtol=1e-5)

    def test_ivar_scaled_consistently(self):
        flux = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]])
        ivar = np.full_like(flux, 4.0)
        norm, scaled = normalize_spectra(flux, ivar)
        # IQR = 2 -> sigma scales by 1/2, ivar by 4
        np.testing.assert_allclose(scaled, 16.0)
        np.testing.assert_allclose(norm, (flux - 2.0) / 2.0)

    def test_all_invalid_row(self):
        flux = np.ones((2, 10))
        ivar = np.ones((2, 10))
        ivar[1] = 0.0
        norm, scaled = normalize_spectra(flux, ivar)
        assert np.all(norm[1] == 0) and np.all(scaled[1] == 0)

    def test_chunking_and_inplace_match(self):
        rng = np.random.default_rng(2)
        flux = rng.normal(size=(37, 50)).astype(np.float32)
        ivar = rng.uniform(0, 2, size=(37, 50)).astype(np.float32)
        ref = normalize_spectra(flux, ivar)
        chunked = normalize_spectra(flux, ivar, chunk_size=5)
        f2, i2 = flux.copy(), ivar.copy()
        out = normalize_spectra(f2, i2, inplace=True)
        for a, b in [(ref, chunked), (ref, out)]:
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])
        assert out[0] is f2 or np.shares_memory(out[0], f2)

    def test_one_dimensional_input(self):
        flux = np.linspace(0, 1, 20)
        norm, scaled = normalize_spectra(flux, np.ones(20))
        assert norm.shape == (20,) and scaled.shape == (20,)


class TestNormalizeSpectrum:
    """Tests for the spectrum normalization function."""

    def test_normalization_zero_median(self):
        """Test that normalized flux has approximately zero median."""
        flux = np.random.randn(1000) * 100 + 500  # Mean ~500
        ivar = np.ones(1000) * 0.01

        norm_flux, _ = normalize_spectrum(flux, ivar)

        # Median should be close to zero
        assert abs(np.median(norm_flux)) < 1e-6

    def test_normalization_unit_iqr(self):
        """Test that normalized flux has IQR close to 1."""
        flux = np.random.randn(1000) * 100 + 500
        ivar = np.ones(1000) * 0.01

        norm_flux, _ = normalize_spectrum(flux, ivar)

        iqr = np.percentile(norm_flux, 75) - np.percentile(norm_flux, 25)
        # IQR should be close to 1
        assert abs(iqr - 1.0) < 0.01

    def test_ivar_scaling(self):
        """Test that IVAR is scaled by IQR^2."""
        flux = np.array([0.0, 1.0, 2.0, 3.0, 4.0])  # IQR = 2.0
        ivar = np.ones(5) * 0.25

        _, scaled_ivar = normalize_spectrum(flux, ivar)

        # IQR = 2.0, so scaling factor is 4.0
        expected_ivar = 4.0 * 0.25
        assert np.allclose(scaled_ivar, expected_ivar)

    def test_nan_ivar_handled(self):
        """Test that NaN values in IVAR are set to zero."""
        flux = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ivar = np.array([1.0, np.nan, 1.0, np.inf, 1.0])

        _, scaled_ivar = normalize_spectrum(flux, ivar)

        # NaN and Inf should become 0
        assert scaled_ivar[1] == 0.0
        assert scaled_ivar[3] == 0.0
        # Normal values should be scaled
        assert scaled_ivar[0] > 0
        assert scaled_ivar[2] > 0

    def test_zero_iqr_handled(self):
        """Test that zero IQR doesn't cause division by zero."""
        # Constant flux has IQR = 0
        flux = np.ones(100) * 5.0
        ivar = np.ones(100) * 0.1

        # Should not raise
        norm_flux, scaled_ivar = normalize_spectrum(flux, ivar)

        # Should have finite values
        assert np.all(np.isfinite(norm_flux))
        assert np.all(np.isfinite(scaled_ivar))


class TestQualityFilter:
    """Tests for the quality filtering function."""

    def test_nan_values_rejected(self):
        """Test that NaN values are flagged as bad quality."""
        labels = np.array([[5000, 2.5, -0.5], [np.nan, 2.5, -0.5]])
        errors = np.ones((2, 3)) * 0.1
        params = ["teff", "logg", "fe_h"]

        mask = apply_quality_filter(labels, errors, params)

        assert mask[0]  # Good
        assert not mask[1]  # Bad (NaN)

    def test_inf_values_rejected(self):
        """Test that Inf values are flagged as bad quality."""
        labels = np.array([[5000, 2.5, -0.5], [5000, np.inf, -0.5]])
        errors = np.ones((2, 3)) * 0.1
        params = ["teff", "logg", "fe_h"]

        mask = apply_quality_filter(labels, errors, params)

        assert mask[0]
        assert not mask[1]

    def test_zero_error_rejected(self):
        """Test that zero errors are flagged as bad quality."""
        labels = np.array([[5000, 2.5, -0.5], [5000, 2.5, -0.5]])
        errors = np.array([[0.1, 0.1, 0.1], [0.1, 0.0, 0.1]])  # Zero error
        params = ["teff", "logg", "fe_h"]

        mask = apply_quality_filter(labels, errors, params)

        assert mask[0]
        assert not mask[1]

    def test_out_of_bounds_rejected(self):
        """Test that values outside physical bounds are rejected."""
        labels = np.array(
            [
                [5000, 2.5, -0.5],  # Good
                [50000, 2.5, -0.5],  # Teff too high (max 10000)
                [5000, 7.0, -0.5],  # logg too high (max 6)
                [5000, 2.5, 5.0],  # [Fe/H] too high (max 1)
            ]
        )
        errors = np.ones((4, 3)) * 0.1
        params = ["teff", "logg", "fe_h"]

        mask = apply_quality_filter(labels, errors, params)

        assert mask[0]  # Good
        assert not mask[1]  # Bad Teff
        assert not mask[2]  # Bad logg
        assert not mask[3]  # Bad [Fe/H]

    def test_zero_values_rejected_except_feh(self):
        """Test that zero values are rejected except for [Fe/H]."""
        labels = np.array(
            [
                [5000, 2.5, 0.0],  # [Fe/H]=0 is valid
                [0, 2.5, -0.5],  # Teff=0 is not valid
            ]
        )
        errors = np.ones((2, 3)) * 0.1
        params = ["teff", "logg", "fe_h"]

        mask = apply_quality_filter(labels, errors, params)

        assert mask[0]  # [Fe/H]=0 allowed
        assert not mask[1]  # Teff=0 not allowed

    def test_all_parameters_checked(self):
        """Test that all 11 parameters have bounds defined."""
        expected_params = list(APOGEE_COLUMN_MAP.keys())

        for param in expected_params:
            assert param in PARAMETER_BOUNDS


class TestSpectralData:
    """Tests for the SpectralData container class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample spectral data for testing."""
        n_samples = 10
        n_wavelengths = 100
        n_params = 3

        flux = np.random.randn(n_samples, n_wavelengths).astype(np.float32)
        ivar = np.abs(np.random.randn(n_samples, n_wavelengths)).astype(np.float32)
        wavelength = np.linspace(3600, 9800, n_wavelengths)
        labels = np.random.randn(n_samples, n_params).astype(np.float32)
        errors = np.abs(np.random.randn(n_samples, n_params)).astype(np.float32) + 0.01
        ids = np.array([f"star_{i}" for i in range(n_samples)])
        quality_mask = np.array(
            [True, True, True, False, True, True, False, True, True, True]
        )

        return SpectralData(
            flux=flux,
            ivar=ivar,
            wavelength=wavelength,
            labels=labels,
            errors=errors,
            ids=ids,
            quality_mask=quality_mask,
            parameter_names=["teff", "logg", "fe_h"],
        )

    def test_properties(self, sample_data):
        """Test that properties return correct values."""
        assert sample_data.n_samples == 10
        assert sample_data.n_wavelengths == 100
        assert sample_data.n_parameters == 3

    def test_get_model_input_shape(self, sample_data):
        """Test that get_model_input returns correct shape."""
        X = sample_data.get_model_input(apply_quality_mask=False)

        assert X.shape == (10, 2, 100)
        assert X.dtype == np.float32

    def test_get_model_input_with_mask(self, sample_data):
        """Test that quality mask is applied correctly."""
        X_masked = sample_data.get_model_input(apply_quality_mask=True)
        X_unmasked = sample_data.get_model_input(apply_quality_mask=False)

        # 2 stars are masked out
        assert X_masked.shape[0] == 8
        assert X_unmasked.shape[0] == 10

    def test_get_labels_shape(self, sample_data):
        """Test that get_labels returns correct shape."""
        labels = sample_data.get_labels(apply_quality_mask=False)
        assert labels.shape == (10, 3)

        labels_masked = sample_data.get_labels(apply_quality_mask=True)
        assert labels_masked.shape == (8, 3)

    def test_get_combined_labels(self, sample_data):
        """Test that combined labels include errors."""
        combined = sample_data.get_combined_labels(apply_quality_mask=False)

        # Should be (n_samples, 2 * n_params)
        assert combined.shape == (10, 6)

        # First half should be labels, second half errors
        assert np.allclose(combined[:, :3], sample_data.labels)
        assert np.allclose(combined[:, 3:], sample_data.errors)


class TestFITSLoader:
    """Tests for the FITSLoader class."""

    @pytest.fixture
    def mock_fits_path(self):
        """Create a temporary mock FITS file for testing."""
        pytest.importorskip("astropy")
        from astropy.io import fits
        from astropy.table import Table

        n_samples = 50
        n_wavelengths = 100

        # Create APOGEE table
        apogee_data = {
            "APOGEE_ID": [f"2M{i:010d}" for i in range(n_samples)],
            "TEFF": np.random.uniform(4000, 6000, n_samples),
            "TEFF_ERR": np.random.uniform(10, 100, n_samples),
            "LOGG": np.random.uniform(2, 5, n_samples),
            "LOGG_ERR": np.random.uniform(0.01, 0.1, n_samples),
            "FE_H": np.random.uniform(-1, 0.5, n_samples),
            "FE_H_ERR": np.random.uniform(0.01, 0.1, n_samples),
            "MG_FE": np.random.uniform(-0.2, 0.4, n_samples),
            "MG_FE_ERR": np.random.uniform(0.01, 0.1, n_samples),
            "C_FE": np.random.uniform(-0.3, 0.3, n_samples),
            "C_FE_ERR": np.random.uniform(0.01, 0.1, n_samples),
            "SI_FE": np.random.uniform(-0.2, 0.3, n_samples),
            "SI_FE_ERR": np.random.uniform(0.01, 0.1, n_samples),
            "NI_FE": np.random.uniform(-0.2, 0.2, n_samples),
            "NI_FE_ERR": np.random.uniform(0.01, 0.1, n_samples),
            "AL_FE": np.random.uniform(-0.3, 0.4, n_samples),
            "AL_FE_ERR": np.random.uniform(0.01, 0.1, n_samples),
            "CA_FE": np.random.uniform(-0.2, 0.3, n_samples),
            "CA_FE_ERR": np.random.uniform(0.01, 0.1, n_samples),
            "N_FE": np.random.uniform(-0.2, 0.5, n_samples),
            "N_FE_ERR": np.random.uniform(0.01, 0.1, n_samples),
            "MN_FE": np.random.uniform(-0.4, 0.2, n_samples),
            "MN_FE_ERR": np.random.uniform(0.01, 0.1, n_samples),
            "ASPCAPFLAG": np.zeros(n_samples, dtype=np.int64),
        }
        apogee_table = Table(apogee_data)

        # Create spectra table
        flux_data = (
            np.random.randn(n_samples, n_wavelengths).astype(np.float32) * 100 + 1000
        )
        spectra_table = Table({"FLUX": list(flux_data)})

        # Create ivar table
        ivar_data = (
            np.abs(np.random.randn(n_samples, n_wavelengths)).astype(np.float32) * 0.01
        )
        ivar_table = Table({"IVAR": list(ivar_data)})

        # Create wavelength table
        wavelength = np.linspace(3600, 9800, n_wavelengths)
        wavelength_table = Table({"WAVELENGTH": wavelength})

        # Write FITS file
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            fits_path = Path(f.name)

        hdu_list = fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.BinTableHDU(apogee_table, name="APOGEE"),
                fits.BinTableHDU(spectra_table, name="SPECTRA"),
                fits.BinTableHDU(ivar_table, name="SPEC_IVAR"),
                fits.BinTableHDU(wavelength_table, name="WAVELENGTH"),
            ]
        )
        hdu_list.writeto(fits_path, overwrite=True)

        yield fits_path

        # Cleanup
        fits_path.unlink()

    def test_loader_creation(self, mock_fits_path):
        """Test that loader can be created."""
        loader = FITSLoader(mock_fits_path)
        assert loader.fits_path == mock_fits_path
        assert len(loader.parameters) == 11

    def test_loader_load_returns_spectral_data(self, mock_fits_path):
        """Test that load returns SpectralData object."""
        loader = FITSLoader(mock_fits_path)
        data = loader.load()

        assert isinstance(data, SpectralData)
        assert data.n_samples == 50
        assert data.n_wavelengths == 100
        assert data.n_parameters == 11

    def test_loader_normalization_applied(self, mock_fits_path):
        """Test that spectrum normalization is applied."""
        loader = FITSLoader(mock_fits_path)
        data = loader.load(apply_normalization=True)

        # Check that flux is normalized (median close to 0)
        for i in range(min(10, data.n_samples)):
            median = np.median(data.flux[i])
            assert abs(median) < 1e-5

    def test_loader_without_normalization(self, mock_fits_path):
        """Test loading without normalization."""
        loader = FITSLoader(mock_fits_path)
        data = loader.load(apply_normalization=False)

        # Raw flux should have larger values (around 1000)
        mean_flux = np.mean(data.flux)
        assert mean_flux > 100

    def test_loader_quality_mask_computed(self, mock_fits_path):
        """Test that quality mask is computed."""
        loader = FITSLoader(mock_fits_path)
        data = loader.load(apply_quality_filter=True)

        # Should have a boolean mask
        assert data.quality_mask.dtype == bool
        assert len(data.quality_mask) == data.n_samples

    def test_loader_file_not_found(self):
        """Test that FileNotFoundError is raised for missing files."""
        loader = FITSLoader(Path("/nonexistent/file.fits"))

        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_loader_custom_parameters(self, mock_fits_path):
        """Test loading with custom parameter subset."""
        loader = FITSLoader(mock_fits_path, parameters=["teff", "logg", "fe_h"])
        data = loader.load()

        assert data.n_parameters == 3
        assert data.parameter_names == ["teff", "logg", "fe_h"]

    def test_ids_extracted(self, mock_fits_path):
        """Test that star IDs are extracted correctly."""
        loader = FITSLoader(mock_fits_path)
        data = loader.load()

        assert len(data.ids) == data.n_samples
        assert all(id.startswith("2M") for id in data.ids)

    @pytest.mark.parametrize("flux_dtype", [np.float32, np.float64])
    def test_rows_match_per_spectrum_normalization(self, tmp_path, flux_dtype):
        """Spectra are normalized in one batch; each row must equal
        normalize_spectrum() applied to that row alone, bit for bit."""
        from astropy.io import fits
        from astropy.table import Table

        from dorothy.data.fits_loader import normalize_spectrum

        rng = np.random.default_rng(0)
        n, n_wave = 40, 33
        flux = rng.normal(1000, 50, (n, n_wave)) * 10.0 ** rng.integers(-2, 3, (n, 1))
        ivar = rng.uniform(0, 0.02, (n, n_wave))
        ivar[rng.random((n, n_wave)) < 0.1] = 0
        flux[rng.random((n, n_wave)) < 0.05] = np.nan
        ivar[3] = 0  # no valid pixels
        flux[4] = 7.0  # zero IQR
        apogee = {c: np.ones(n) for c in ("TEFF", "TEFF_ERR", "LOGG", "LOGG_ERR")}
        path = tmp_path / "spectra.fits"
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.BinTableHDU(Table(apogee), name="APOGEE"),
                fits.BinTableHDU(
                    Table({"FLUX": flux.astype(flux_dtype)}), name="SPECTRA"
                ),
                fits.BinTableHDU(
                    Table({"IVAR": ivar.astype(np.float32)}), name="SPEC_IVAR"
                ),
            ]
        ).writeto(path)

        data = FITSLoader(path, parameters=["teff", "logg"]).load()
        for i in range(n):
            row_flux, row_ivar = normalize_spectrum(
                flux[i].astype(flux_dtype).astype(np.float32),
                ivar[i].astype(np.float32),
            )
            np.testing.assert_array_equal(data.flux[i], row_flux)
            np.testing.assert_array_equal(data.ivar[i], row_ivar)

    @pytest.mark.parametrize("normalize", [False, True])
    def test_variable_length_columns_load_like_fixed_length_columns(
        self, tmp_path, normalize
    ):
        from astropy.io import fits

        rng = np.random.default_rng(0)
        flux = rng.normal(1000, 50, (5, 12)).astype(np.float32)
        ivar = rng.uniform(0.01, 0.02, (5, 12)).astype(np.float32)
        flux[0, 3], ivar[1, 4], ivar[2, 5] = np.nan, np.nan, -1.0
        loaded = []
        for name, fmt in [("fixed", "12E"), ("variable", "PE()")]:
            path = tmp_path / f"{name}.fits"
            fits.HDUList(
                [fits.PrimaryHDU()]
                + [
                    fits.BinTableHDU.from_columns(
                        [fits.Column(column, fmt, array=array)], name=ext
                    )
                    for column, ext, array in [
                        ("FLUX", "SPECTRA", flux),
                        ("IVAR", "SPEC_IVAR", ivar),
                    ]
                ]
            ).writeto(path)
            loaded.append(
                FITSLoader(path).load(
                    apply_normalization=normalize,
                    apply_quality_filter=False,
                    require_labels=False,
                )
            )
        fixed, variable = loaded
        np.testing.assert_array_equal(variable.flux, fixed.flux)
        np.testing.assert_array_equal(variable.ivar, fixed.ivar)
        assert fixed.flux[0, 3] == fixed.ivar[1, 4] == fixed.ivar[2, 5] == 0


class TestSplitData:
    """Tests for the data splitting function."""

    @pytest.fixture
    def sample_spectral_data(self):
        """Create sample data for splitting tests."""
        n_samples = 100
        n_wavelengths = 50
        n_params = 3

        return SpectralData(
            flux=np.random.randn(n_samples, n_wavelengths).astype(np.float32),
            ivar=np.abs(np.random.randn(n_samples, n_wavelengths)).astype(np.float32),
            wavelength=np.linspace(3600, 9800, n_wavelengths),
            labels=np.random.randn(n_samples, n_params).astype(np.float32),
            errors=np.abs(np.random.randn(n_samples, n_params)).astype(np.float32)
            + 0.01,
            ids=np.array([f"star_{i}" for i in range(n_samples)]),
            quality_mask=np.ones(n_samples, dtype=bool),
            parameter_names=["teff", "logg", "fe_h"],
        )

    def test_split_ratios(self, sample_spectral_data):
        """Test that splits have correct ratios."""
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(
            sample_spectral_data,
            train_ratio=0.7,
            val_ratio=0.2,
        )

        assert X_train.shape[0] == 70
        assert X_val.shape[0] == 20
        assert X_test.shape[0] == 10

    def test_split_shapes(self, sample_spectral_data):
        """Test that split data has correct shapes."""
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(
            sample_spectral_data
        )

        # X should be (n, 2, wavelengths)
        assert X_train.shape[1:] == (2, 50)
        assert X_val.shape[1:] == (2, 50)
        assert X_test.shape[1:] == (2, 50)

        # y should be (n, 2 * n_params)
        assert y_train.shape[1] == 6
        assert y_val.shape[1] == 6
        assert y_test.shape[1] == 6

    def test_split_reproducibility(self, sample_spectral_data):
        """Test that same seed produces same splits."""
        split1 = split_data(sample_spectral_data, seed=42)
        split2 = split_data(sample_spectral_data, seed=42)

        (X_train1, _), _, _ = split1
        (X_train2, _), _, _ = split2

        assert np.allclose(X_train1, X_train2)

    def test_split_different_seeds(self, sample_spectral_data):
        """Test that different seeds produce different splits."""
        (X_train1, _), _, _ = split_data(sample_spectral_data, seed=42)
        (X_train2, _), _, _ = split_data(sample_spectral_data, seed=123)

        # Should not be equal
        assert not np.allclose(X_train1, X_train2)

    def test_split_no_overlap(self, sample_spectral_data):
        """Test that train/val/test sets don't overlap."""
        (X_train, _), (X_val, _), (X_test, _) = split_data(sample_spectral_data)

        total = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
        assert total == 100  # All samples accounted for

    def test_split_with_quality_mask(self, sample_spectral_data):
        """Test splitting with quality mask applied."""
        # Mask out 20 samples
        sample_spectral_data.quality_mask[:20] = False

        (X_train, _), (X_val, _), (X_test, _) = split_data(
            sample_spectral_data,
            apply_quality_mask=True,
        )

        total = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
        assert total == 80  # Only 80 good samples


class TestColumnMappings:
    """Tests for column name mappings."""

    def test_all_parameters_have_mappings(self):
        """Test that all 11 parameters have column mappings."""
        expected_params = [
            "teff",
            "logg",
            "fe_h",
            "mg_fe",
            "c_fe",
            "si_fe",
            "ni_fe",
            "al_fe",
            "ca_fe",
            "n_fe",
            "mn_fe",
        ]

        for param in expected_params:
            assert param in APOGEE_COLUMN_MAP
            label_col, err_col = APOGEE_COLUMN_MAP[param]
            assert isinstance(label_col, str)
            assert isinstance(err_col, str)
            assert err_col.endswith("_ERR")

    def test_all_parameters_have_bounds(self):
        """Test that all 11 parameters have physical bounds."""
        expected_params = list(APOGEE_COLUMN_MAP.keys())

        for param in expected_params:
            assert param in PARAMETER_BOUNDS
            low, high = PARAMETER_BOUNDS[param]
            assert low < high


class TestIvarToErrorAndMask:
    """Tests for the ivar_to_error_and_mask function."""

    def test_basic_conversion(self):
        """Test basic ivar to error conversion."""
        ivar = np.array([4.0, 1.0, 0.25], dtype=np.float32)

        error, mask = ivar_to_error_and_mask(ivar)

        # error = 1/sqrt(ivar)
        assert np.allclose(error, [0.5, 1.0, 2.0])
        assert np.allclose(mask, [1.0, 1.0, 1.0])

    def test_nan_handling(self):
        """Test that NaN values are masked."""
        ivar = np.array([1.0, np.nan, 1.0], dtype=np.float32)

        error, mask = ivar_to_error_and_mask(ivar)

        assert error[1] == 0.0
        assert mask[1] == 0.0
        assert mask[0] == 1.0
        assert mask[2] == 1.0

    def test_inf_handling(self):
        """Test that Inf values are masked."""
        ivar = np.array([1.0, np.inf, -np.inf], dtype=np.float32)

        error, mask = ivar_to_error_and_mask(ivar)

        assert error[1] == 0.0
        assert error[2] == 0.0
        assert mask[1] == 0.0
        assert mask[2] == 0.0

    def test_zero_handling(self):
        """Test that zero ivar is masked."""
        ivar = np.array([1.0, 0.0, 1.0], dtype=np.float32)

        error, mask = ivar_to_error_and_mask(ivar)

        assert error[1] == 0.0
        assert mask[1] == 0.0

    def test_negative_handling(self):
        """Test that negative ivar is masked."""
        ivar = np.array([1.0, -1.0, 1.0], dtype=np.float32)

        error, mask = ivar_to_error_and_mask(ivar)

        assert error[1] == 0.0
        assert mask[1] == 0.0

    def test_2d_array(self):
        """Test with 2D array (batch of spectra)."""
        ivar = np.array([[4.0, 0.0, 1.0], [1.0, np.nan, 0.25]], dtype=np.float32)

        error, mask = ivar_to_error_and_mask(ivar)

        assert error.shape == (2, 3)
        assert mask.shape == (2, 3)
        # Check first row
        assert np.allclose(error[0], [0.5, 0.0, 1.0])
        assert np.allclose(mask[0], [1.0, 0.0, 1.0])
        # Check second row
        assert np.allclose(error[1], [1.0, 0.0, 2.0])
        assert np.allclose(mask[1], [1.0, 0.0, 1.0])


class TestSpectralData3Channel:
    """Tests for 3-channel model input."""

    @pytest.fixture
    def sample_data_3channel(self):
        """Create sample data with spectral mask and error."""
        n_samples = 10
        n_wavelengths = 100
        n_params = 3

        flux = np.random.randn(n_samples, n_wavelengths).astype(np.float32)
        ivar = (
            np.abs(np.random.randn(n_samples, n_wavelengths)).astype(np.float32) + 0.1
        )

        # Create some masked wavelengths
        ivar[:, :10] = 0.0  # First 10 wavelengths masked for all stars

        spectral_error, spectral_mask = ivar_to_error_and_mask(ivar)

        return SpectralData(
            flux=flux,
            ivar=ivar,
            wavelength=np.linspace(3600, 9800, n_wavelengths),
            labels=np.random.randn(n_samples, n_params).astype(np.float32),
            errors=np.abs(np.random.randn(n_samples, n_params)).astype(np.float32)
            + 0.01,
            ids=np.array([f"star_{i}" for i in range(n_samples)]),
            quality_mask=np.ones(n_samples, dtype=bool),
            parameter_names=["teff", "logg", "fe_h"],
            spectral_mask=spectral_mask,
            spectral_error=spectral_error,
        )

    def test_get_model_input_3channel_shape(self, sample_data_3channel):
        """Test that get_model_input_3channel returns correct shape."""
        X = sample_data_3channel.get_model_input_3channel(apply_quality_mask=False)

        assert X.shape == (10, 3, 100)
        assert X.dtype == np.float32

    def test_get_model_input_3channel_channels(self, sample_data_3channel):
        """Test that 3-channel output has correct channel content."""
        X = sample_data_3channel.get_model_input_3channel(apply_quality_mask=False)

        # Channel 0 is flux
        assert np.allclose(X[:, 0, :], sample_data_3channel.flux)
        # Channel 1 is error
        assert np.allclose(X[:, 1, :], sample_data_3channel.spectral_error)
        # Channel 2 is mask
        assert np.allclose(X[:, 2, :], sample_data_3channel.spectral_mask)

    def test_get_model_input_3channel_mask_values(self, sample_data_3channel):
        """Test that mask channel has expected values."""
        X = sample_data_3channel.get_model_input_3channel(apply_quality_mask=False)

        # First 10 wavelengths should be masked (0)
        assert np.all(X[:, 2, :10] == 0.0)
        # Rest should be valid (1)
        assert np.all(X[:, 2, 10:] == 1.0)

    def test_get_model_input_3channel_without_mask_raises(self):
        """Test that 3-channel raises if spectral_mask not computed."""
        data = SpectralData(
            flux=np.random.randn(5, 50).astype(np.float32),
            ivar=np.random.randn(5, 50).astype(np.float32),
            wavelength=np.linspace(3600, 9800, 50),
            labels=np.random.randn(5, 3).astype(np.float32),
            errors=np.random.randn(5, 3).astype(np.float32),
            ids=np.array([f"star_{i}" for i in range(5)]),
            quality_mask=np.ones(5, dtype=bool),
            parameter_names=["teff", "logg", "fe_h"],
        )

        with pytest.raises(ValueError, match="3-channel format requires"):
            data.get_model_input_3channel()


class TestFITSLoaderSpectralMask:
    """Tests for FITSLoader with spectral mask computation."""

    @pytest.fixture
    def mock_fits_path_with_masked_ivar(self):
        """Create a mock FITS file with some masked (zero) ivar values."""
        pytest.importorskip("astropy")
        from astropy.io import fits
        from astropy.table import Table

        n_samples = 20
        n_wavelengths = 100

        # Create APOGEE table
        apogee_data = {
            "APOGEE_ID": [f"2M{i:010d}" for i in range(n_samples)],
            "TEFF": np.random.uniform(4000, 6000, n_samples),
            "TEFF_ERR": np.random.uniform(10, 100, n_samples),
            "LOGG": np.random.uniform(2, 5, n_samples),
            "LOGG_ERR": np.random.uniform(0.01, 0.1, n_samples),
            "FE_H": np.random.uniform(-1, 0.5, n_samples),
            "FE_H_ERR": np.random.uniform(0.01, 0.1, n_samples),
        }
        apogee_table = Table(apogee_data)

        # Create spectra table
        flux_data = (
            np.random.randn(n_samples, n_wavelengths).astype(np.float32) * 100 + 1000
        )
        spectra_table = Table({"FLUX": list(flux_data)})

        # Create ivar table with some zeros
        ivar_data = (
            np.abs(np.random.randn(n_samples, n_wavelengths)).astype(np.float32) * 0.01
        )
        ivar_data[:, :20] = 0.0  # First 20 wavelengths masked
        ivar_table = Table({"IVAR": list(ivar_data)})

        wavelength_table = Table({"WAVELENGTH": np.linspace(3600, 9800, n_wavelengths)})

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            primary = fits.PrimaryHDU()
            apogee_hdu = fits.BinTableHDU(apogee_table, name="APOGEE")
            spectra_hdu = fits.BinTableHDU(spectra_table, name="SPECTRA")
            ivar_hdu = fits.BinTableHDU(ivar_table, name="SPEC_IVAR")
            wavelength_hdu = fits.BinTableHDU(wavelength_table, name="WAVELENGTH")

            hdul = fits.HDUList(
                [primary, apogee_hdu, spectra_hdu, ivar_hdu, wavelength_hdu]
            )
            hdul.writeto(f.name, overwrite=True)

            yield Path(f.name)

        Path(f.name).unlink()

    def test_load_with_spectral_mask(self, mock_fits_path_with_masked_ivar):
        """Test loading with compute_spectral_mask=True."""
        loader = FITSLoader(
            mock_fits_path_with_masked_ivar, parameters=["teff", "logg", "fe_h"]
        )
        data = loader.load(compute_spectral_mask=True)

        assert data.spectral_mask is not None
        assert data.spectral_error is not None
        assert data.spectral_mask.shape == data.flux.shape
        assert data.spectral_error.shape == data.flux.shape

    def test_spectral_mask_values(self, mock_fits_path_with_masked_ivar):
        """Test that spectral mask has correct values."""
        loader = FITSLoader(
            mock_fits_path_with_masked_ivar, parameters=["teff", "logg", "fe_h"]
        )
        data = loader.load(compute_spectral_mask=True)

        # First 20 wavelengths should be masked
        assert np.all(data.spectral_mask[:, :20] == 0.0)
        # Rest should be valid
        assert np.all(data.spectral_mask[:, 20:] == 1.0)

    def test_flux_zeroed_at_masked_wavelengths(self, mock_fits_path_with_masked_ivar):
        """Test that flux is zeroed at masked wavelengths."""
        loader = FITSLoader(
            mock_fits_path_with_masked_ivar, parameters=["teff", "logg", "fe_h"]
        )
        data = loader.load(compute_spectral_mask=True)

        # Flux should be zero where mask is zero
        assert np.all(data.flux[:, :20] == 0.0)

    def test_3channel_output_from_loader(self, mock_fits_path_with_masked_ivar):
        """Test getting 3-channel output from loaded data."""
        loader = FITSLoader(
            mock_fits_path_with_masked_ivar, parameters=["teff", "logg", "fe_h"]
        )
        data = loader.load(compute_spectral_mask=True)

        X = data.get_model_input_3channel()
        assert X.shape[1] == 3  # 3 channels

    def test_load_without_spectral_mask_default(self, mock_fits_path_with_masked_ivar):
        """Test that spectral_mask is None by default."""
        loader = FITSLoader(
            mock_fits_path_with_masked_ivar, parameters=["teff", "logg", "fe_h"]
        )
        data = loader.load(compute_spectral_mask=False)

        assert data.spectral_mask is None
        assert data.spectral_error is None


def test_precision_encoding_replaces_only_valid_sigma_and_leaves_input_unchanged():
    """sigma becomes log(1 + 1/sigma) where the mask is positive and 0 where
    it is not, whatever a masked pixel holds; flux, mask and the input are
    unchanged."""
    nan, inf = float("nan"), float("inf")
    x = np.array(
        [
            [
                [1.0, -2.0, 3.0, 4.0, 5.0, 6.0],
                [0.5, 4.0, 0.0, nan, inf, -1.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=np.float32,
    )
    original = x.copy()

    encoded = encode_precision(x)

    expected = np.array([np.log(3.0), np.log(1.25), 0, 0, 0, 0], dtype=np.float32)
    np.testing.assert_allclose(encoded[0, 1], expected, rtol=1e-6)
    np.testing.assert_array_equal(encoded[:, [0, 2]], x[:, [0, 2]])
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("sigma", [0.0, -1.0, float("nan")])
def test_precision_encoding_rejects_a_valid_pixel_without_positive_sigma(sigma):
    x = np.ones((1, 3, 4), dtype=np.float32)
    x[0, 1, 2] = sigma
    with pytest.raises(ValueError, match="sigma > 0"):
        encode_precision(x)
