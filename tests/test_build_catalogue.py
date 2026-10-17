"""End-to-end tests for scripts/build_catalogue_v2.py on synthetic FITS inputs."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from dorothy.data import CatalogueLoader

SCRIPT = Path(__file__).parent.parent / "scripts" / "build_catalogue_v2.py"
N_STARS = 8
N_WAVE = 40


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location("build_catalogue_v2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _table(name, columns):
    cols = []
    for col_name, data in columns.items():
        data = np.asarray(data)
        if data.ndim == 1:
            fmt = "K" if data.dtype.kind == "i" else "D"
            cols.append(fits.Column(name=col_name, format=fmt, array=data))
        else:
            cols.append(
                fits.Column(name=col_name, format=f"{data.shape[1]}E", array=data)
            )
    return fits.BinTableHDU.from_columns(cols, name=name)


def _write_pair(path, gaia_ids, label_hdu, rng):
    flux = (rng.normal(size=(len(gaia_ids), N_WAVE)) + 10).astype(np.float32)
    ivar = np.ones_like(flux)
    hdus = [
        fits.PrimaryHDU(),
        _table("GAIA_ID", {"GAIA_ID": np.asarray(gaia_ids, dtype=np.int64)}),
        _table("WAVELENGTH", {"WAVELENGTH": np.linspace(3600, 9000, N_WAVE)}),
        _table("SPECTRA", {"FLUX": flux}),
        _table("SPEC_IVAR", {"IVAR": ivar}),
        label_hdu,
    ]
    fits.HDUList(hdus).writeto(path)


def _apogee_hdu(rng, n):
    cols = {
        "TEFF": rng.uniform(4000, 6000, n),
        "TEFF_ERR": np.full(n, 50.0),
        "LOGG": rng.uniform(1, 5, n),
        "LOGG_ERR": np.full(n, 0.05),
        "FE_H": rng.normal(0, 0.3, n),
        "FE_H_ERR": np.full(n, 0.02),
        "FE_H_FLAG": np.zeros(n, dtype=np.int64),
        "ASPCAPFLAG": np.zeros(n, dtype=np.int64),
        "X_H": rng.normal(0, 0.3, (n, 24)).astype(np.float32),
        "X_H_ERR": np.full((n, 24), 0.03, dtype=np.float32),
    }
    for el in ["MG", "C", "SI", "NI", "AL", "CA", "N", "MN", "O", "NA", "K"]:
        cols[f"{el}_FE"] = rng.normal(0, 0.1, n)
        cols[f"{el}_FE_ERR"] = np.full(n, 0.04)
        cols[f"{el}_FE_FLAG"] = np.zeros(n, dtype=np.int64)
    # TI, V, CR, CO, BA, SR deliberately absent as [X/Fe] columns
    return _table("APOGEE", cols)


def _galah_hdu(rng, n):
    cols = {
        "teff": rng.uniform(4000, 6000, n),
        "e_teff": np.full(n, 80.0),
        "logg": rng.uniform(1, 5, n),
        "e_logg": np.full(n, 0.1),
        "fe_h": rng.normal(0, 0.3, n),
        "e_fe_h": np.full(n, 0.03),
        "flag_fe_h": np.array([0] * (n - 1) + [1], dtype=np.int64),
        "flag_sp": np.zeros(n, dtype=np.int64),
    }
    for el in ["mg", "c", "si", "ni", "al", "ca", "mn", "o", "na", "k", "ti", "ba"]:
        cols[f"{el}_fe"] = rng.normal(0, 0.1, n)
        cols[f"e_{el}_fe"] = np.full(n, 0.04)
        cols[f"flag_{el}_fe"] = np.zeros(n, dtype=np.int64)
    return _table("GALAH", cols)


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory):
    rng = np.random.default_rng(0)
    d = tmp_path_factory.mktemp("raw")
    apogee_ids = np.arange(1, N_STARS + 1)
    galah_ids = np.arange(5, N_STARS + 5)  # partial overlap with APOGEE
    _write_pair(
        d / "BOSSxAPOGEE_training.fits", apogee_ids, _apogee_hdu(rng, N_STARS), rng
    )
    _write_pair(
        d / "BOSSxGALAH_training.fits", galah_ids, _galah_hdu(rng, N_STARS), rng
    )
    return d


def _labels(path, source):
    import h5py

    with h5py.File(path, "r") as f:
        grp = f["labels"][source]
        return grp["values"][:], grp["errors"][:], grp["flags"][:]


def test_default_build_has_11_xfe_parameters(builder, raw_dir, tmp_path):
    out = tmp_path / "cat.h5"
    builder.build_catalogue(raw_dir, out, verbose=False)
    loader = CatalogueLoader(out)
    assert loader.get_parameter_names() == builder.PARAMETER_NAMES
    assert len(builder.PARAMETER_NAMES) == 11
    values, _, _ = _labels(out, "apogee")
    assert values.shape[1] == 11


def test_extended_xh_build(builder, raw_dir, tmp_path):
    out = tmp_path / "cat_xh.h5"
    elements = builder.EXTENDED_ELEMENTS
    builder.build_catalogue(
        raw_dir, out, verbose=False, elements=elements, abundance_scale="x_h"
    )
    names = CatalogueLoader(out).get_parameter_names()
    assert len(names) == 20
    assert names[:4] == ["teff", "logg", "fe_h", "mg_h"]
    assert names[-1] == "sr_h"

    import h5py

    with h5py.File(out, "r") as f:
        gaia = f["metadata"]["gaia_id"][:]
        assert f.attrs["abundance_scale"] == "x_h"

    with fits.open(raw_dir / "BOSSxGALAH_training.fits") as hdul:
        galah = hdul["GALAH"].data
        galah_ids = hdul["GAIA_ID"].data["GAIA_ID"]
    with fits.open(raw_dir / "BOSSxAPOGEE_training.fits") as hdul:
        apogee = hdul["APOGEE"].data
        apogee_ids = hdul["GAIA_ID"].data["GAIA_ID"]

    # GALAH [Mg/H] = [Mg/Fe] + [Fe/H], errors in quadrature
    g_vals, g_errs, g_flags = _labels(out, "galah")
    rows = np.searchsorted(gaia, galah_ids)
    mg = names.index("mg_h")
    np.testing.assert_allclose(
        g_vals[rows, mg], galah["mg_fe"] + galah["fe_h"], rtol=1e-5
    )
    np.testing.assert_allclose(g_errs[rows, mg], np.hypot(0.04, 0.03), rtol=1e-5)
    # Bad [Fe/H] propagates to derived [X/H]
    assert g_flags[rows[-1], mg] == 1 and g_flags[rows[0], mg] == 0
    # Elements GALAH lacks (sr) are flagged
    assert np.all(g_flags[rows, names.index("sr_h")] == 1)

    # APOGEE [X/H] comes straight from the X_H array (Mg is index 5)
    a_vals, a_errs, a_flags = _labels(out, "apogee")
    rows = np.searchsorted(gaia, apogee_ids)
    np.testing.assert_allclose(a_vals[rows, mg], apogee["X_H"][:, 5], rtol=1e-5)
    np.testing.assert_allclose(a_errs[rows, mg], 0.03, rtol=1e-5)
    # Ti has no TI_FE column but is in X_H (index 12): still filled
    ti = names.index("ti_h")
    np.testing.assert_allclose(a_vals[rows, ti], apogee["X_H"][:, 12], rtol=1e-5)
    # Ba/Sr are not APOGEE elements: flagged, so the loader masks them
    assert np.all(a_flags[rows, names.index("ba_h")] == 1)

    data = CatalogueLoader(out).load_merged_sparse(
        ["boss"], label_sources=["apogee", "galah"], max_flag_bits=-1
    )
    assert data.n_params == 20
    apogee_mask = data.labels_dict["apogee"][:, 2, :]
    assert apogee_mask[:, names.index("ba_h")].sum() == 0
    assert apogee_mask[:, mg].sum() == N_STARS


def test_invalid_abundance_scale(builder):
    with pytest.raises(ValueError):
        builder.get_parameter_names(abundance_scale="x_y")


def _snr_reference(flux, ivar):
    """The original per-spectrum SNR: median of |flux| sqrt(ivar), valid pixels."""
    snr = np.zeros(len(flux), dtype=np.float32)
    for i in range(len(flux)):
        valid = ivar[i] > 0
        if valid.sum() > 0:
            snr[i] = np.median(np.abs(flux[i, valid]) * np.sqrt(ivar[i, valid]))
    return snr


def _edge_spectra(rng, n, n_wave):
    flux = (
        rng.normal(10, 3, (n, n_wave)) * 10.0 ** rng.integers(-2, 3, (n, 1))
    ).astype(np.float32)
    ivar = rng.uniform(0, 2, (n, n_wave)).astype(np.float32)
    ivar[rng.random((n, n_wave)) < 0.1] = 0
    ivar[rng.random((n, n_wave)) < 0.02] = -1
    ivar[rng.random((n, n_wave)) < 0.02] = np.nan
    flux[rng.random((n, n_wave)) < 0.01] = np.nan
    flux[rng.random((n, n_wave)) < 0.01] = np.inf
    ivar[0] = 0  # no valid pixels
    ivar[1, 1:] = 0  # one valid pixel
    ivar[2, 2:] = 0  # two valid pixels
    flux[3, :2], ivar[3, :2] = 0.0, np.inf  # 0 * inf = NaN (sign differs from flux NaN)
    flux[3, 2] = np.nan
    return flux, ivar


@pytest.mark.parametrize("chunk_rows", [1, 7, 256])
def test_median_snr_matches_per_spectrum_median(builder, chunk_rows):
    rng = np.random.default_rng(3)
    for n_wave in (1, 2, 5, 40, 501):
        flux, ivar = _edge_spectra(rng, 60, max(n_wave, 3))
        with np.errstate(invalid="ignore", over="ignore"):
            expected = _snr_reference(flux, ivar)
            got = builder.median_snr(flux, ivar, chunk_rows=chunk_rows)
        assert got.dtype == np.float32
        assert got.tobytes() == expected.tobytes()  # bitwise, NaNs included


def test_build_with_edge_case_spectra(builder, tmp_path):
    """SNR and spectra placement on files with bad pixels, overlapping IDs,
    invalid/duplicate IDs and the LAMOST MRS dual-arm layout."""
    rng = np.random.default_rng(4)
    raw = tmp_path / "raw"
    raw.mkdir()
    n = 30
    boss_ids = {
        "apogee": np.arange(1, n + 1),
        "galah": np.arange(20, n + 20),  # 11 overlap with APOGEE
    }
    boss_ids["apogee"][5] = 0  # invalid
    boss_ids["galah"][3] = boss_ids["galah"][4]  # duplicate within a file
    boss = {}
    for source, ids in boss_ids.items():
        flux, ivar = _edge_spectra(rng, n, N_WAVE)
        boss[source] = (ids, flux, ivar)
        label_hdu = (_apogee_hdu if source == "apogee" else _galah_hdu)(rng, n)
        hdus = [
            fits.PrimaryHDU(),
            _table("GAIA_ID", {"GAIA_ID": ids.astype(np.int64)}),
            _table("WAVELENGTH", {"WAVELENGTH": np.linspace(3600, 9000, N_WAVE)}),
            _table("SPECTRA", {"FLUX": flux}),
            _table("SPEC_IVAR", {"IVAR": ivar}),
            label_hdu,
        ]
        name = f"BOSSx{source.upper()}_training.fits"
        fits.HDUList(hdus).writeto(raw / name)
    n_b, n_r = 12, 9
    mrs_ids = np.arange(100, 100 + n)
    fb, ib = _edge_spectra(rng, n, n_b)
    fr, ir = _edge_spectra(rng, n, n_r)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            _table("GAIA_ID", {"GAIA_ID": mrs_ids}),
            _table("WAVELENGTH_B", {"WAVELENGTH_B": np.linspace(4950, 5350, n_b)}),
            _table("WAVELENGTH_R", {"WAVELENGTH_R": np.linspace(6300, 6800, n_r)}),
            _table("SPECTRA_B", {"FLUX_B": fb}),
            _table("SPEC_IVAR_B", {"IVAR_B": ib}),
            _table("SPECTRA_R", {"FLUX_R": fr}),
            _table("SPEC_IVAR_R", {"IVAR_R": ir}),
            _apogee_hdu(rng, n),
        ]
    ).writeto(raw / "LAMOST_MRSxAPOGEE.fits")

    out = tmp_path / "cat.h5"
    with np.errstate(invalid="ignore", over="ignore"):
        builder.build_catalogue(raw, out, verbose=False)

    import h5py

    with h5py.File(out, "r") as f:
        gaia = f["metadata"]["gaia_id"][:]
        snr = {s: f["surveys"][s]["snr"][:] for s in ("boss", "lamost_mrs")}
        boss_flux = f["surveys"]["boss"]["flux"][:]
        spectra = f["surveys"]["lamost_mrs"]["spectra"][:]

    # APOGEE rows first, then GALAH fills stars not yet present
    expected_snr = np.zeros(len(gaia), dtype=np.float32)
    expected_flux = np.zeros((len(gaia), N_WAVE), dtype=np.float32)
    filled = np.zeros(len(gaia), dtype=bool)
    for source in ("apogee", "galah"):
        ids, flux, ivar = boss[source]
        with np.errstate(invalid="ignore", over="ignore"):
            ref = _snr_reference(flux, ivar)
        rows = np.flatnonzero(ids > 0)
        target = np.searchsorted(gaia, ids[rows])
        new = ~filled[target]
        expected_snr[target[new]] = ref[rows[new]]
        expected_flux[target[new]] = flux[rows[new]]
        filled[target[new]] = True
    assert snr["boss"].tobytes() == expected_snr.tobytes()
    np.testing.assert_array_equal(boss_flux, expected_flux)

    target = np.searchsorted(gaia, mrs_ids)
    with np.errstate(invalid="ignore", over="ignore"):
        ref = _snr_reference(fb, ib)  # blue arm
    assert snr["lamost_mrs"][target].tobytes() == ref.tobytes()
    np.testing.assert_array_equal(spectra[target, 0, :n_b], fb)
    np.testing.assert_array_equal(spectra[target, 3, :n_r], ir)
    assert np.all(spectra[target, 2, n_r:] == 0)  # red arm padding
