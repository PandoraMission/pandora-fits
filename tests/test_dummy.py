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


def test_create_dummy_data_N0():
    level = logger.level
    logger.setLevel("ERROR")
    NIRDALevel0HDUList().writeto(
        f"{TESTDIR}dummyfiles/nirda-level0.fits", overwrite=True
    )
    logger.setLevel(level)


def test_create_dummy_data_N1():
    level = logger.level
    logger.setLevel("ERROR")
    NIRDALevel1HDUList().writeto(
        f"{TESTDIR}dummyfiles/nirda-level1.fits", overwrite=True
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


def test_create_dummy_data_V1():
    level = logger.level
    logger.setLevel("ERROR")
    VISDALevel1HDUList().writeto(
        f"{TESTDIR}dummyfiles/visda-level1.fits", overwrite=True
    )
    logger.setLevel(level)


def test_create_dummy_data_V2():
    level = logger.level
    logger.setLevel("ERROR")
    VISDALevel2HDUList().writeto(
        f"{TESTDIR}dummyfiles/visda-level2.fits", overwrite=True
    )
    logger.setLevel(level)


def test_create_dummy_data_E0():
    level = logger.level
    logger.setLevel("ERROR")
    EngineeringLevel0HDUList().writeto(
        f"{TESTDIR}dummyfiles/eng-level0.fits", overwrite=True
    )
    logger.setLevel(level)


def test_create_dummy_data_L3():
    level = logger.level
    logger.setLevel("ERROR")
    Level3HDUList().writeto(f"{TESTDIR}dummyfiles/level3.fits", overwrite=True)
    logger.setLevel(level)
