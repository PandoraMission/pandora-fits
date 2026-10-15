# Standard library
import warnings

# Third-party
from astropy.io import fits

# First-party/Local
from pandorafits import logger
from pandorafits.nirda import NIRDALevel0HDUList


def stale_dummy_file():
    """Dummy file with a wrong NEXTEND and checksums that do not match."""
    level = logger.level
    logger.setLevel("ERROR")
    hdulist = NIRDALevel0HDUList()
    logger.setLevel(level)
    for hdu in hdulist:
        hdu.header["CHECKSUM"] = "stalestalestale0"
        hdu.header["DATASUM"] = "0"
    hdulist[0].header["NEXTEND"] = 99
    return hdulist


def test_writeto_refreshes_checksums_and_nextend(tmp_path):
    path = tmp_path / "file.fits"
    stale_dummy_file().writeto(str(path))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with fits.open(path, checksum=True) as hdulist:
            assert hdulist[0].header["NEXTEND"] == len(hdulist) - 1
            assert all("CHECKSUM" in hdu.header for hdu in hdulist)


def test_writeto_without_checksum_drops_stale_cards(tmp_path):
    path = tmp_path / "file.fits"
    stale_dummy_file().writeto(str(path), checksum=False)
    with fits.open(path) as hdulist:
        for hdu in hdulist:
            assert "CHECKSUM" not in hdu.header
            assert "DATASUM" not in hdu.header
