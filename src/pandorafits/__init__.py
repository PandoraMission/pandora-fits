__version__ = "0.1.0"
import logging  # noqa: E402
import os  # noqa

import numpy as np  # noqa
import pandas as pd  # noqa
import pandoraref as pr

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
FORMATSDIR = f"{PACKAGEDIR}/formats/"
logger = logging.getLogger("pandorafits")


NIRDAReference = pr.NIRDAReference()
VISDAReference = pr.VISDAReference()


from .nirda import *  # noqa
from .visda import *  # noqa
