"""Class to handle Pandora fits files"""

import numpy as np
import pandas as pd
from .fits import PandoraHDUList, FORMATSDIR

__all__ = [
    "NIRDALevel0HDUList",
    "NIRDALevel1HDUList",
    "NIRDALevel2HDUList",
]


class NIRDALevel0HDUList(PandoraHDUList):
    """NIRDA Level 0 File Type"""

    def __init__(self, file=None):
        """
        This class will read a file passed to it using astropy fits.
        After loading it will validate that the file is compliant with
        the NIRDA Level 0 file standards specified in the `formats`
        folder. This object subclasses the fits.HDUList object, and
        maintains all its class methods.

        Parameters:
        -----------
        file: None, str, fits.HDUList
            The file to load
        """
        self.header_formats = [
            pd.read_excel(FORMATSDIR + "nirda/level0-headers.xlsx", idx)
            for idx in range(2)
        ]
        self.extension_types = pd.read_excel(
            FORMATSDIR + "nirda/level0-extension-types.xlsx"
        )
        super().__init__(file=file)


class NIRDALevel1HDUList(PandoraHDUList):
    def __init__(self, file=None):
        self.header_formats = [
            pd.read_excel(FORMATSDIR + "nirda/level1-headers.xlsx", idx)
            for idx in range(2)
        ]
        self.extension_types = pd.read_excel(
            FORMATSDIR + "nirda/level1-extension-types.xlsx"
        )
        super().__init__(file=file)


class NIRDALevel2HDUList(PandoraHDUList):
    def __init__(self, file=None):
        self.header_formats = [
            pd.read_excel(FORMATSDIR + "nirda/level2-headers.xlsx", idx)
            for idx in range(7)
        ]
        self.extension_types = pd.read_excel(
            FORMATSDIR + "nirda/level2-extension-types.xlsx"
        )
        super().__init__(file=file)
