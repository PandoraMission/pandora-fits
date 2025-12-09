import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

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

    def to_level1(self):
        return VISDAFFILevel1HDUList(self)


class VISDAFFILevel1HDUList(VISDAFFILevel0HDUList):
    filename = FORMATSDIR + "visda/level1-ffi_visda.xlsx"
    reference = VISDAReference

    def to_level1(self):
        raise ValueError("This is a level 1 product.")


class VISDALevel0HDUList(PandoraHDUList):
    filename = FORMATSDIR + "visda/level0_visda.xlsx"
    reference = VISDAReference

    @property
    def coord(self):
        return SkyCoord(
            self[0].header["TARG_RA"], self[0].header["TARG_DEC"], unit="deg"
        )

    @property
    def ROI_corners(self):
        roistrtx = self[0].header["ROISTRTX"]
        roistrty = self[0].header["ROISTRTY"]
        return [
            (y + roistrty, x + roistrtx)
            for x, y in Table(self[2].data).to_pandas().values
        ]

    @property
    def ROI_size(self):
        return (self[0].header["STARDIMS"], self[0].header["STARDIMS"])

    @property
    def nROI(self):
        return self[0].header["NUMSTARS"]

    @property
    def data_cube(self):
        numSubFrms = int(np.ceil(np.sqrt(self.nROI)))
        dims = (numSubFrms * self.ROI_size[0], numSubFrms * self.ROI_size[1])
        if self[1].header["NAXIS1"] == dims[0]:
            ex = 0
        elif self[1].header["NAXIS1"] == (dims[0] + numSubFrms + 1):
            ex = 1
        else:
            raise ValueError("Can not parse the padding dimensions.")
        data = self[1].data[:, ex:, ex:]
        shape = data.shape[1:]
        width = self.ROI_size[0]
        nims = int(shape[1] / (width + ex))
        d1 = np.asarray(np.array_split(data, nims, axis=2))
        nims = int(shape[0] / (width + ex))
        d2 = np.asarray(np.array_split(d1, nims, axis=2))[
            :, :, :, :-ex, :-ex
        ].transpose([2, 0, 1, 3, 4])
        return d2

    @property
    def star_list(self):
        d = self.data_cube[:, ::-1]
        starlist = []
        numSubFrms = int(np.ceil(np.sqrt(self.nROI)))
        for idx in range(numSubFrms):
            for jdx in range(numSubFrms):
                if (idx * numSubFrms + jdx) == self.nROI:
                    return np.asarray(starlist).transpose([1, 0, 2, 3])
                starlist.append(d[:, idx, jdx])

    @property
    def star_row(self):
        return np.asarray(
            [r + np.arange(self.ROI_size[0]) for r, _ in self.ROI_corners]
        )

    @property
    def star_column(self):
        return np.asarray(
            [c + np.arange(self.ROI_size[1]) for _, c in self.ROI_corners]
        )

    def to_level1(self):
        return VISDALevel1HDUList(self)


class VISDALevel1HDUList(VISDALevel0HDUList):
    filename = FORMATSDIR + "visda/level1_visda.xlsx"
    reference = VISDAReference

    def to_level1(self):
        raise ValueError("This is a level 1 product.")


class VISDALevel2HDUList(PandoraHDUList):
    filename = FORMATSDIR + "visda/level2_visda.xlsx"
    reference = VISDAReference
