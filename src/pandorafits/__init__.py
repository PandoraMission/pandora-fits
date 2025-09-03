__version__ = "0.1.0"
import logging  # noqa: E402
import os  # noqa

import numpy as np  # noqa
import pandas as pd  # noqa

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
FORMATSDIR = f"{PACKAGEDIR}/formats/"
logger = logging.getLogger("pandorafits")


from .fits import EngineeringLevel0HDUList  # noqa
from .fits import Level3HDUList  # noqa
from .nirda import NIRDALevel0HDUList  # noqa
from .nirda import NIRDALevel1HDUList  # noqa
from .nirda import NIRDALevel2HDUList  # noqa
from .visda import VISDAFFILevel0HDUList  # noqa
from .visda import VISDALevel0HDUList  # noqa
from .visda import VISDALevel1HDUList  # noqa
from .visda import VISDALevel2HDUList  # noqa
