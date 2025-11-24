from astropy.coordinates import SkyCoord

from . import FORMATSDIR, VISDAReference
from .fits import PandoraHDUList

__all__ = [
    "VISDAFFILevel0HDUList",
    "VISDAFFILevel1HDUList",
    "VISDALevel0HDUList",
    "VISDALevel1HDUList",
    "VISDALevel2HDUList",
]


class VISDAFFILevel0HDUList(PandoraHDUList):
    filename = FORMATSDIR + "visda/level0-ffi_visda.xlsx"
    reference = VISDAReference


class VISDAFFILevel1HDUList(PandoraHDUList):
    filename = FORMATSDIR + "visda/level1-ffi_visda.xlsx"
    reference = VISDAReference


class VISDALevel0HDUList(PandoraHDUList):
    filename = FORMATSDIR + "visda/level0_visda.xlsx"
    reference = VISDAReference


class VISDALevel1HDUList(PandoraHDUList):
    filename = FORMATSDIR + "visda/level1_visda.xlsx"
    reference = VISDAReference

    @property
    def coord(self):
        return SkyCoord(
            self[0].header["TARG_RA"], self[0].header["TARG_DEC"], unit="deg"
        )


class VISDALevel2HDUList(PandoraHDUList):
    filename = FORMATSDIR + "visda/level2_visda.xlsx"
    reference = VISDAReference
