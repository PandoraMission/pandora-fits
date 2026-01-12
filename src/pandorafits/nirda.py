import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord

from . import FORMATSDIR, NIRDAReference, logger
from .fits import PandoraHDUList
from .io import register_hdulist
from .scene import get_NIRDA_scene

__all__ = [
    "NIRDALevel0HDUList",
    "NIRDALevel1HDUList",
    "NIRDALevel2HDUList",
]


@register_hdulist(
    lambda h: h
    and (
        (h[0].header.get("TELESCOP") == "NASA Pandora")
        & (h[0].header.get("INSTRMNT") == "NIRDA")
        & ("PFCLASS" not in h[0].header)
    )
)
class NIRDALevel0HDUList(PandoraHDUList):
    filename = FORMATSDIR + "nirda/level0_nirda.xlsx"
    reference = NIRDAReference
    level = 0
    instrument = "NIRDA"

    def __to_l1__(self):
        return NIRDALevel1HDUList(self)

    def plot_data(self, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots()
        d = self["science"].data[0]
        k = d != 0
        vmin = kwargs.pop("vmin", np.nanpercentile(d[k], 1))
        vmax = kwargs.pop("vmax", np.nanpercentile(d[k], 1) + 100)
        im = ax.pcolormesh(self.column, self.row, d, vmin=vmin, vmax=vmax, **kwargs)
        ax.set(
            aspect="equal",
            title=f"{self[0].header['targ_id']} {self.start_time.isot}",
            xlabel="ROI Column",
            ylabel="ROI Row",
        )
        plt.colorbar(im, ax=ax)
        # ax.margins(0)
        return ax


@register_hdulist(
    lambda h: h
    and (
        (h[0].header.get("TELESCOP") == "NASA Pandora")
        & (h[0].header.get("INSTRMNT") == "NIRDA")
        & (h[0].header.get("PFCLASS") == "NIRDALevel1HDUList")
    )
)
class NIRDALevel1HDUList(NIRDALevel0HDUList):
    filename = FORMATSDIR + "nirda/level1_nirda.xlsx"
    level = 1

    # def get_scene(self):
    #     hdr = self[0].header
    #     prf = pa.DispersedPRF.from_reference()
    #     prf.imcorner = (hdr["ROISTRTY"], hdr["ROISTRTX"])
    #     prf.imshape = (hdr["ROISIZEY"], hdr["ROISIZEX"])
    #     scene = pa.DispersedSkyScene(prf, self.wcs, self.start_time)
    #     return scene

    def get_scene(self):
        hdr = self[0].header
        imcorner = (hdr["ROISTRTY"], hdr["ROISTRTX"])
        imshape = (hdr["ROISIZEY"], hdr["ROISIZEX"])
        return get_NIRDA_scene(
            time_jd=self.sequence_start_time.jd,
            ra=hdr["TARG_RA"],
            dec=hdr["TARG_DEC"],
            roll=hdr["TARG_RLL"],
            imcorner=imcorner,
            imshape=imshape,
        )

    def _append_scene_extensions(self):
        hdr = self[0].header
        scene = self.get_scene()
        self.append(scene.get_catalog_hdu())
        self.append(scene.get_prf_hdu())
        self.append(scene.get_model_hdu())
        self.append(
            scene.get_aperture_hdu(
                SkyCoord(hdr["targ_ra"], hdr["targ_dec"], unit="deg"),
                relative_threshold=0.0001,
                absolute_threshold=30,
            )
        )
        self[0].header["GAIA_ID"] = self["APERTURE"].header["GAIA_ID"]
        logger.info("Appended scene extensions")

    def __to_l2__(self):
        return NIRDALevel2HDUList(self)


@register_hdulist(
    lambda h: h
    and (
        (h[0].header.get("TELESCOP") == "NASA Pandora")
        & (h[0].header.get("INSTRMNT") == "NIRDA")
        & (h[0].header.get("PFCLASS") == "NIRDALevel2HDUList")
    )
)
class NIRDALevel2HDUList(NIRDALevel1HDUList):
    filename = FORMATSDIR + "nirda/level2_nirda.xlsx"
    level = 2
