from . import FORMATSDIR, NIRDAReference
from .fits import PandoraHDUList

__all__ = [
    "NIRDAFFILevel0HDUList",
    "NIRDAFFILevel1HDUList",
    "NIRDALevel0HDUList",
    "NIRDALevel1HDUList",
    "NIRDALevel2HDUList",
]


class NIRDAFFILevel0HDUList(PandoraHDUList):
    filename = FORMATSDIR + "nirda/level0-ffi_nirda.xlsx"
    reference = NIRDAReference


class NIRDAFFILevel1HDUList(PandoraHDUList):
    filename = FORMATSDIR + "nirda/level1-ffi_nirda.xlsx"
    reference = NIRDAReference


class NIRDALevel0HDUList(PandoraHDUList):
    filename = FORMATSDIR + "nirda/level0_nirda.xlsx"
    reference = NIRDAReference


class NIRDALevel1HDUList(PandoraHDUList):
    filename = FORMATSDIR + "nirda/level1_nirda.xlsx"
    reference = NIRDAReference


class NIRDALevel2HDUList(PandoraHDUList):
    filename = FORMATSDIR + "nirda/level2_nirda.xlsx"
    reference = NIRDAReference
