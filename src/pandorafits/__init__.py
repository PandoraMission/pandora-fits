__version__ = "0.1.0"
import logging  # noqa: E402
import os  # noqa
import pandas as pd  # noqa
import numpy as np  # noqa

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
FORMATSDIR = f"{PACKAGEDIR}/formats/"
logger = logging.getLogger("pandorafits")

# BITPIX_DICT = {
#     8: np.uint8,
#     16: np.int16,
#     32: np.float32,
#     -32: np.float32,
#     -64: np.float64,
# }
BITPIX_DICT = {
    8: ">u1",  # Unsigned 8-bit integer, big-endian
    16: ">i2",  # Signed 16-bit integer, big-endian
    32: ">i4",  # Signed 32-bit integer, big-endian
    -32: ">f4",  # 32-bit floating-point (float32), big-endian
    -64: ">f8",  # 64-bit floating-point (float64), big-endian
}


from .fits import (  # noqa
    NIRDALevel0HDUList,  # noqa
    NIRDALevel2HDUList,  # noqa
    VISDALevel0HDUList,  # noqa
    VISDALevel2HDUList,  # noqa
)
