# First-party/Local
import os
from pathlib import Path

from pandorafits import PACKAGEDIR, logger
from pandorafits.nirda import NIRDALevel0HDUList
from pandorafits.visda import VISDAFFILevel0HDUList, VISDALevel0HDUList

TESTDIR = str(Path(PACKAGEDIR).resolve().parents[1] / "tests") + os.sep


def test_create_dummy_data_N0():
    level = logger.level
    logger.setLevel("ERROR")
    NIRDALevel0HDUList().writeto(
        f"{TESTDIR}dummyfiles/nirda-level0.fits", overwrite=True
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
    VISDAFFILevel0HDUList().writeto(
        f"{TESTDIR}dummyfiles/visda-level1.fits", overwrite=True
    )
    logger.setLevel(level)
