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
