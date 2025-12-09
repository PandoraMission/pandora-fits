import os

import pandorafits.database as pfdb  # noqa
from pandorafits import PACKAGEDIR, logger
from pandorafits.nirda import NIRDAFFILevel0HDUList, NIRDALevel0HDUList
from pandorafits.visda import VISDAFFILevel0HDUList, VISDALevel0HDUList

TESTDIR = "/".join(PACKAGEDIR.split("/")[:-2]) + "/tests/"


def test_roundtrip():
    level = logger.level
    logger.setLevel("ERROR")
    for HDUList in [
        NIRDALevel0HDUList,
        VISDALevel0HDUList,
        NIRDAFFILevel0HDUList,
        VISDAFFILevel0HDUList,
    ]:
        dummy_hdulist = HDUList()
        dummy_hdulist.writeto("test.fits", overwrite=True)
        HDUList("test.fits")
    os.remove("test.fits")
    logger.setLevel(level)
