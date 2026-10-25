"""Write the inputs of the quick start: a synthetic catalogue, a config and spectra.

Run it from the repository root::

    python examples/synthetic_quickstart.py

It writes three files into ``quickstart/``:

- ``catalogue.h5``: a synthetic super-catalogue of 2000 stars, with BOSS,
  DESI and LAMOST MRS spectra and APOGEE and GALAH labels. The labels are
  smooth functions of hidden variables that also generate the spectra, so a
  network can learn them.
- ``config.yaml``: a small model that reads BOSS and DESI spectra and
  predicts APOGEE labels.
- ``boss_spectra.fits``: the BOSS spectra of the stars that have no APOGEE
  labels, in the format that ``dorothy predict`` reads.
"""

import sys
from pathlib import Path

import h5py
from astropy.io import fits

OUT = Path("quickstart")


def write_fits(catalogue: Path, path: Path) -> int:
    """Write the BOSS spectra of the stars without APOGEE labels to FITS."""
    with h5py.File(catalogue, "r") as f:
        boss = f["surveys/boss"]
        unlabelled = (boss["snr"][:] > 0) & (f["labels/apogee/errors"][:] == 0).all(1)
        flux = boss["flux"][:][unlabelled]
        ivar = boss["ivar"][:][unlabelled]
        gaia_id = f["metadata/gaia_id"][:][unlabelled]
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU.from_columns(
                [fits.Column("GAIA_ID", "K", array=gaia_id)], name="GAIA_ID"
            ),
            fits.BinTableHDU.from_columns(
                [fits.Column("FLUX", f"{flux.shape[1]}E", array=flux)], name="SPECTRA"
            ),
            fits.BinTableHDU.from_columns(
                [fits.Column("IVAR", f"{ivar.shape[1]}E", array=ivar)],
                name="SPEC_IVAR",
            ),
        ]
    ).writeto(path, overwrite=True)
    return len(flux)


def main() -> None:
    # The catalogue generator lives in the test suite, which pip does not install
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tests.synthetic import build_catalogue, write_config

    OUT.mkdir(exist_ok=True)
    catalogue = build_catalogue(OUT / "catalogue.h5", n_stars=2000)
    config = write_config(
        OUT / "config.yaml",
        catalogue,
        OUT / "outputs",
        surveys=["boss", "desi"],
        label_sources=["apogee"],
        name="quickstart",
        epochs=30,
    )
    n_spectra = write_fits(catalogue, OUT / "boss_spectra.fits")
    print(f"Wrote {catalogue}, {config} and {n_spectra} spectra to")
    print(f"{OUT / 'boss_spectra.fits'}")


if __name__ == "__main__":
    main()
