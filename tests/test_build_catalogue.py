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
    # Elements GALAH lacks (sr) are masked via error 0, not flagged
    assert np.all(g_flags[rows, names.index("sr_h")] == 0)
    assert np.all(g_errs[rows, names.index("sr_h")] == 0)

    # APOGEE [X/H] comes straight from the X_H array (Mg is index 5)
    a_vals, a_errs, a_flags = _labels(out, "apogee")
    rows = np.searchsorted(gaia, apogee_ids)
    np.testing.assert_allclose(a_vals[rows, mg], apogee["X_H"][:, 5], rtol=1e-5)
    np.testing.assert_allclose(a_errs[rows, mg], 0.03, rtol=1e-5)
    # Ti has no TI_FE column but is in X_H (index 12): still filled
    ti = names.index("ti_h")
    np.testing.assert_allclose(a_vals[rows, ti], apogee["X_H"][:, 12], rtol=1e-5)
    # Ba/Sr are not APOGEE elements: masked via error 0, not flagged
    assert np.all(a_flags[rows, names.index("ba_h")] == 0)
    assert np.all(a_errs[rows, names.index("ba_h")] == 0)

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


def test_unmeasured_elements_do_not_reject_stars(builder, raw_dir, tmp_path):
    """With max_flag_bits=0, a source lacking some elements must keep its stars."""
    out = tmp_path / "cat_ext.h5"
    builder.build_catalogue(
        raw_dir,
        out,
        verbose=False,
        elements=builder.EXTENDED_ELEMENTS,
        abundance_scale="x_h",
    )
    data = CatalogueLoader(out).load_merged_sparse(
        ["boss"], label_sources=["apogee", "galah"], max_flag_bits=0
    )
    for source in ["apogee", "galah"]:
        mask = data.labels_dict[source][:, 2, :]
        assert mask[:, 0].sum() > 0, source  # teff survives the flag filter


def test_xh_without_fe_h_is_masked(builder):
    """Derived [X/H] needs [Fe/H]; without it the abundances stay masked."""
    import numpy.lib.recfunctions as rfn

    n = 4
    data = np.zeros(
        n, dtype=[("teff", "f8"), ("e_teff", "f8"), ("mg_fe", "f8"), ("e_mg_fe", "f8")]
    )
    data["teff"], data["e_teff"], data["mg_fe"], data["e_mg_fe"] = 5000, 50, 0.1, 0.05
    data = rfn.repack_fields(data)
    values, errors, flags = builder.extract_galah_labels(data, ["mg"], "x_h")
    assert np.all(errors[:, 3] == 0) and np.all(flags[:, 3] == 0)


def test_select_best_prefers_higher_score_then_earlier(builder):
    targets = np.array([3, 1, 3, 1, 2])
    scores = np.array([5.0, 2.0, 7.0, 2.0, 0.0])
    current = np.array([0.0, 0.0, 0.0, 6.0])
    # Row 1: tie -> first candidate; row 2: score 0 doesn't beat 0;
    # row 3: 7 beats 5 and the current 6
    assert sorted(builder.select_best(targets, scores, current)) == [1, 2]


def test_spectrum_snr_ignores_masked_and_empty_rows(builder):
    flux = np.array([[2.0, 4.0, 100.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    ivar = np.array([[1.0, 4.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(builder.spectrum_snr(flux, ivar), [5.0, 0.0])


def _write_spectra_file(path, gaia_ids, flux, ivar, label_hdu):
    hdus = [
        fits.PrimaryHDU(),
        _table("GAIA_ID", {"GAIA_ID": np.asarray(gaia_ids, dtype=np.int64)}),
        _table("WAVELENGTH", {"WAVELENGTH": np.linspace(3600, 9000, N_WAVE)}),
        _table("SPECTRA", {"FLUX": flux}),
        _table("SPEC_IVAR", {"IVAR": ivar}),
        label_hdu,
    ]
    fits.HDUList(hdus).writeto(path)


def _nan_rows(hdu, rows):
    """Copy of a label table with every float column NaN in the given rows."""
    data = hdu.data.copy()
    for name in data.dtype.names:
        if data[name].dtype.kind == "f":
            data[name][rows] = np.nan
    return fits.BinTableHDU(data, name=hdu.name)


def test_bad_first_rows_do_not_block_later_files(builder, tmp_path):
    """An unusable first match (NaN labels / spectrum) must not lock out later ones."""
    rng = np.random.default_rng(1)
    raw = tmp_path / "raw"
    raw.mkdir()
    ids = np.arange(1, N_STARS + 1)
    flux = (rng.normal(size=(N_STARS, N_WAVE)) + 10).astype(np.float32)
    ivar = np.ones_like(flux)

    # BOSS x APOGEE: star 1 has NaN labels and an all-NaN spectrum
    bad_flux = flux.copy()
    bad_flux[0] = np.nan
    boss_apogee = _nan_rows(_apogee_hdu(rng, N_STARS), [0])
    _write_spectra_file(
        raw / "BOSSxAPOGEE_training.fits", ids, bad_flux, ivar, boss_apogee
    )
    # BOSS x GALAH: star 1 again, with a valid spectrum; star 2 observed
    # twice, the repeat with 4x the inverse variance
    galah_ids = np.concatenate([ids, [2]])
    galah_flux = np.concatenate([flux, flux[1:2] + 1]).astype(np.float32)
    galah_ivar = np.ones_like(galah_flux)
    galah_ivar[-1] = 4.0
    _write_spectra_file(
        raw / "BOSSxGALAH_training.fits",
        galah_ids,
        galah_flux,
        galah_ivar,
        _galah_hdu(rng, N_STARS + 1),
    )
    # LAMOST LRS x APOGEE: valid labels for star 1
    lrs_apogee = _apogee_hdu(rng, N_STARS)
    _write_spectra_file(
        raw / "LAMOST_LRSxAPOGEE_training.fits", ids, flux, ivar, lrs_apogee
    )

    out = tmp_path / "cat.h5"
    builder.build_catalogue(raw, out, verbose=False)
    import h5py

    with h5py.File(out, "r") as f:
        row = {g: i for i, g in enumerate(f["metadata"]["gaia_id"][:])}
        boss_flux = f["surveys"]["boss"]["flux"][:]
        boss_ivar = f["surveys"]["boss"]["ivar"][:]
        boss_snr = f["surveys"]["boss"]["snr"][:]
        teff = f["labels"]["apogee"]["values"][:, 0]

    # Star 1: spectrum from the GALAH file, labels from the LRS file
    np.testing.assert_allclose(boss_flux[row[1]], flux[0])
    assert boss_snr[row[1]] > 0
    assert teff[row[1]] == pytest.approx(lrs_apogee.data["TEFF"][0], rel=1e-6)
    # Other stars keep the first file's labels
    assert teff[row[3]] == pytest.approx(boss_apogee.data["TEFF"][2], rel=1e-6)
    # Star 2: the higher-S/N repeat observation is kept
    np.testing.assert_allclose(boss_flux[row[2]], flux[1] + 1)
    np.testing.assert_allclose(boss_ivar[row[2]], 4.0)
    assert np.isfinite(boss_flux).all()


def test_lamost_mrs_snr_uses_both_arms(builder, tmp_path):
    rng = np.random.default_rng(2)
    raw = tmp_path / "raw"
    raw.mkdir()
    n, n_b, n_r = 3, 10, 12
    ids = np.arange(1, n + 1)
    flux_b = (rng.normal(size=(n, n_b)) + 10).astype(np.float32)
    flux_r = (rng.normal(size=(n, n_r)) + 10).astype(np.float32)
    ivar_b = np.ones_like(flux_b)
    ivar_r = np.ones_like(flux_r)
    ivar_b[0] = 0.0  # star 1: blue arm entirely masked
    hdus = [
        fits.PrimaryHDU(),
        _table("GAIA_ID", {"GAIA_ID": ids}),
        _table("WAVELENGTH_B", {"WAVELENGTH_B": np.linspace(4950, 5350, n_b)}),
        _table("WAVELENGTH_R", {"WAVELENGTH_R": np.linspace(6300, 6800, n_r)}),
        _table("SPECTRA_B", {"FLUX_B": flux_b}),
        _table("SPEC_IVAR_B", {"IVAR_B": ivar_b}),
        _table("SPECTRA_R", {"FLUX_R": flux_r}),
        _table("SPEC_IVAR_R", {"IVAR_R": ivar_r}),
        _apogee_hdu(rng, n),
    ]
    fits.HDUList(hdus).writeto(raw / "LAMOST_MRSxAPOGEE.fits")

    out = tmp_path / "cat.h5"
    builder.build_catalogue(raw, out, verbose=False)
    import h5py

    with h5py.File(out, "r") as f:
        snr = f["surveys"]["lamost_mrs"]["snr"][:]
        spectra = f["surveys"]["lamost_mrs"]["spectra"][:]
    assert np.all(snr > 0)
    np.testing.assert_allclose(snr[0], np.median(np.abs(flux_r[0])), rtol=1e-6)
    np.testing.assert_allclose(spectra[0, 2, :n_r], flux_r[0])
