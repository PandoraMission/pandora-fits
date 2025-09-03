import os

from pandorafits import PACKAGEDIR, logger
from pandorafits.fits import (
    EngineeringLevel0HDUList,
    Level3HDUList,
    NIRDALevel0HDUList,
    NIRDALevel1HDUList,
    NIRDALevel2HDUList,
    VISDALevel0HDUList,
    VISDALevel1HDUList,
    VISDALevel2HDUList,
)

TESTDIR = "/".join(PACKAGEDIR.split("/")[:-2]) + "/tests/"


def test_roundtrip():
    level = logger.level
    logger.setLevel("ERROR")
    for HDUList in [
        NIRDALevel0HDUList,
        NIRDALevel1HDUList,
        NIRDALevel2HDUList,
        VISDALevel0HDUList,
        VISDALevel1HDUList,
        VISDALevel2HDUList,
        EngineeringLevel0HDUList,
        Level3HDUList,
    ]:
        dummy_hdulist = HDUList()
        dummy_hdulist.writeto("test.fits", overwrite=True)
        HDUList("test.fits")
    os.remove("test.fits")
    logger.setLevel(level)
