from pandorafits.fits import (
    NIRDALevel0HDUList,
    NIRDALevel2HDUList,
    VISDALevel0HDUList,
    VISDALevel2HDUList,
)
from pandorafits import PACKAGEDIR
from pandorafits import logger

TESTDIR = "/".join(PACKAGEDIR.split("/")[:-2]) + "/tests/"


def test_create_dummy_data_N0():
    level = logger.level
    logger.setLevel("ERROR")
    NIRDALevel0HDUList().writeto(
        f"{TESTDIR}dummyfiles/nirda-level0.fits", overwrite=True
    )
    logger.setLevel(level)


def test_create_dummy_data_N2():
    level = logger.level
    logger.setLevel("ERROR")
    NIRDALevel2HDUList().writeto(
        f"{TESTDIR}dummyfiles/nirda-level2.fits", overwrite=True
    )
    logger.setLevel(level)


def test_create_dummy_data_V0():
    level = logger.level
    logger.setLevel("ERROR")
    VISDALevel0HDUList().writeto(
        f"{TESTDIR}dummyfiles/visda-level0.fits", overwrite=True
    )
    logger.setLevel(level)


def test_create_dummy_data_V2():
    level = logger.level
    logger.setLevel("ERROR")
    VISDALevel2HDUList().writeto(
        f"{TESTDIR}dummyfiles/visda-level2.fits", overwrite=True
    )
    logger.setLevel(level)
