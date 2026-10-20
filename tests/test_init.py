# Standard library
import os
from pathlib import Path

# First-party/Local
import pandorafits.database as pfdb  # noqa
from pandorafits import PACKAGEDIR, logger
from pandorafits.nirda import NIRDALevel0HDUList
from pandorafits.visda import VISDAFFILevel0HDUList, VISDALevel0HDUList

TESTDIR = str(Path(PACKAGEDIR).resolve().parents[1] / "tests") + os.sep


def test_roundtrip():
    level = logger.level
    logger.setLevel("ERROR")
    for HDUList in [
        NIRDALevel0HDUList,
        VISDALevel0HDUList,
        VISDAFFILevel0HDUList,
    ]:
        dummy_hdulist = HDUList()
        dummy_hdulist.writeto("test.fits", overwrite=True)
        HDUList("test.fits")
    os.remove("test.fits")
    logger.setLevel(level)


def test_time_metadata():
    level = logger.level
    logger.setLevel("ERROR")
    for HDUList in [
        NIRDALevel0HDUList,
        VISDALevel0HDUList,
        VISDAFFILevel0HDUList,
    ]:
        # Check generated headers before validation can fill missing cards.
        hdulist = HDUList().to_level1(upcast=False)
        assert hdulist["TIME"].header["FRAME"] == "JD"
        assert hdulist["TIME"].header["TIMESYS"] == "UTC"
        if HDUList is NIRDALevel0HDUList:
            hdulist.difference_sample()
            assert hdulist["TIME"].header["TIMESYS"] == "UTC"
    logger.setLevel(level)
