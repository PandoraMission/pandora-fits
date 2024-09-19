from pandorafits.fits import (
    NIRDALevel0HDUList,
    NIRDALevel2HDUList,
    VISDALevel0HDUList,
    VISDALevel2HDUList,
)
from pandorafits import PACKAGEDIR
from pandorafits import logger
import os

TESTDIR = "/".join(PACKAGEDIR.split("/")[:-2]) + "/tests/"


def test_roundtrip():
    level = logger.level
    logger.setLevel("ERROR")
    for HDUList in [
        NIRDALevel0HDUList,
        NIRDALevel2HDUList,
        VISDALevel0HDUList,
        VISDALevel2HDUList,
    ]:
        dummy_hdulist = HDUList()
        dummy_hdulist.writeto("test.fits", overwrite=True)
        HDUList("test.fits")
    os.remove("test.fits")
    logger.setLevel(level)
