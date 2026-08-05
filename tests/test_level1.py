# Standard library
import os
from pathlib import Path

# Third-party
import numpy as np

# First-party/Local
import pandorafits as pf
from pandorafits import PACKAGEDIR, logger

TESTDIR = str(Path(PACKAGEDIR).resolve().parents[1] / "tests") + os.sep


def test_bias_subtraction_visimg():
    level = logger.level
    fname = str(
        Path(PACKAGEDIR).resolve().parents[1]
        / "tests"
        / "testdata"
        / "tiny_visimg.fits"
    )
    hdulist = pf.open(fname)
    med = np.median(hdulist["SCIENCE"].data)
    assert np.isclose(med, 112, atol=20)
    hdulist._subtract_bias()

    med = np.median(hdulist["SCIENCE"].data)
    assert np.isclose(med, 0, atol=20)
    logger.setLevel(level)


def test_bias_subtraction_vissci():
    level = logger.level
    fname = str(
        Path(PACKAGEDIR).resolve().parents[1]
        / "tests"
        / "testdata"
        / "tiny_vissci.fits"
    )
    hdulist = pf.open(fname)
    k = ~hdulist.border
    med = np.median(hdulist["SCIENCE"].data[k])
    assert np.isclose(med, 112 * hdulist.frmpcoad, atol=20 * hdulist.frmpcoad)
    hdulist._subtract_bias()

    med = np.median(hdulist["SCIENCE"].data[k])
    assert np.isclose(med, 0, atol=20 * hdulist.frmpcoad)
    logger.setLevel(level)
