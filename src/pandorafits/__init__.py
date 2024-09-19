__version__ = "0.1.0"
import logging  # noqa: E402
import os  # noqa
import pandas as pd  # noqa
import numpy as np  # noqa

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
FORMATSDIR = f"{PACKAGEDIR}/fileformats/"
logger = logging.getLogger("pandorafits")

BITPIX_DICT = {
    8: np.uint8,
    16: np.int16,
    32: np.float32,
    -32: np.float32,
    -64: np.float64,
}


from .fits import NIRDA0HDUList, NIRDA2HDUList, VISDA0HDUList, VISDA2HDUList  # noqa
