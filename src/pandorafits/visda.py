"""Class to handle Pandora fits files"""

import numpy as np
import pandas as pd
from .fits import PandoraHDUList, FORMATSDIR


__all__ = [
    "VISDAFFILevel0HDUList",
    "VISDALevel0HDUList",
    "VISDALevel1HDUList",
    "VISDALevel2HDUList",
]


class VISDAFFILevel0HDUList(PandoraHDUList):
    def __init__(self, file=None):
        self.extension_types = pd.read_excel(
            FORMATSDIR + "visda/level0-ffi-headers.xlsx", 0
        )
        self.header_formats = [
            pd.read_excel(FORMATSDIR + "visda/level0-ffi-headers.xlsx", idx + 1)
            for idx in range(len(self.extension_types))
        ]
        super().__init__(file=file)


class VISDALevel0HDUList(PandoraHDUList):
    def __init__(self, file=None):
        self.extension_types = pd.read_excel(
            FORMATSDIR + "visda/level0-extension-types.xlsx"
        )
        self.header_formats = [
            pd.read_excel(FORMATSDIR + "visda/level0-headers.xlsx", idx)
            for idx in range(len(self.extension_types))
        ]
        super().__init__(file=file)


class VISDALevel1HDUList(PandoraHDUList):
    def __init__(self, file=None):
        self.header_formats = [
            pd.read_excel(FORMATSDIR + "visda/level1-headers.xlsx", idx)
            for idx in range(3)
        ]
        self.extension_types = pd.read_excel(
            FORMATSDIR + "visda/level1-extension-types.xlsx"
        )
        super().__init__(file=file)


class VISDALevel2HDUList(PandoraHDUList):
    def __init__(self, file=None, nROIs: int = 9):
        self.header_formats = [
            pd.read_excel(FORMATSDIR + "visda/level2-headers.xlsx", 0),
            *[
                pd.read_excel(FORMATSDIR + "visda/level2-headers.xlsx", 1)
                for _ in range(1, nROIs + 1)
            ],
            *[
                pd.read_excel(FORMATSDIR + "visda/level2-headers.xlsx", idx)
                for idx in range(2, 5)
            ],
        ]
        self.extension_types = pd.read_excel(
            FORMATSDIR + "visda/level2-extension-types.xlsx"
        )
        self.header_formats[1].loc[
            self.header_formats[1].Name == "EXTNAME", "Value"
        ] = "TARGET"
        for idx in np.arange(2, nROIs + 2):
            self.header_formats[idx].loc[
                self.header_formats[1].Name == "EXTNAME", "Value"
            ] = f"STAR{idx - 1:03}"
            print(self.header_formats[idx])
        super().__init__(file=file)
