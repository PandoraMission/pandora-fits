# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from . import FORMATSDIR, NIRDAReference, logger
from .fits import PandoraHDUList
from .io import register_hdulist
from .report import ReportMixins
from .scene import get_NIRDA_scene
from .utils import convert_time

__all__ = [
    "NIRDALevel0HDUList",
    "NIRDALevel1HDUList",
    "NIRDALevel2HDUList",
]


def get_nirda_frame_indexes_tot(hdr):
    totframes = 0
    times = []
    for e in range(hdr["EXPOSRES"]):
        for i in range(hdr["INTEGRTS"]):
            if i == 0:
                totframes += hdr["RESETS1"]
            else:
                totframes += hdr["RESETS2"]
            totframes += hdr["DROPS1"]
            for g in range(hdr["GRPS"]):
                if g != 0:
                    totframes += hdr["DROPS2"]
                times.extend(totframes + np.arange(hdr["READS"]))
                totframes += hdr["READS"]
                if g == hdr["GRPS"] - 1:
                    totframes += hdr["DROPS3"]
    return np.asarray(times)


def get_nirda_frame_times(hdr):
    indexes = get_nirda_frame_indexes_tot(hdr)
    return convert_time(hdr["CORSTIME"], hdr["FINETIME"]) + (
        indexes * hdr["FRMTIME"] * u.millisecond
    )


def get_nirda_read_frame_indexes(hdr):
    totframes = 0
    times = []
    for e in range(hdr["EXPOSRES"]):
        for i in range(hdr["INTEGRTS"]):
            for g in range(hdr["GRPS"]):
                times.extend(totframes + np.arange(hdr["READS"]))
                totframes += hdr["READS"]
    return np.asarray(times)


def get_nirda_acqd_ints(hdr0):
    if hdr0["FRMSACQ"] != hdr0["FRMSEXP"]:
        idxs = get_nirda_read_frame_indexes(hdr0)
        idxs = idxs.reshape((hdr0["INTEGRTS"], hdr0["GRPS"], hdr0["READS"]))
        acqd_reads = idxs < (hdr0["FRMSACQ"] - 1)
        acqd_ints = np.where(acqd_reads.all(axis=(1, 2)))[0][-1]
    else:
        acqd_ints = hdr0["INTEGRTS"]
    return acqd_ints


def get_nirda_exposure_times(hdr):
    times = []
    for e in range(hdr["EXPOSRES"]):
        for i in range(hdr["INTEGRTS"]):
            totframes = 0
            for g in range(hdr["GRPS"]):
                if g != 0:
                    totframes += hdr["DROPS2"]
                times.extend(totframes + np.arange(1, hdr["READS"] + 1))
                totframes += hdr["READS"]
    return np.asarray(times)


@register_hdulist(
    lambda h: (
        h
        and (
            (h[0].header.get("TELESCOP") == "NASA Pandora")
            & (h[0].header.get("INSTRMNT") == "NIRDA")
            & ("PFCLASS" not in h[0].header)
        )
    )
)
class NIRDALevel0HDUList(PandoraHDUList, ReportMixins):
    filename = FORMATSDIR + "nirda/level0_nirda.xlsx"
    reference = NIRDAReference
    level = 0
    instrument = "NIRDA"

    def __to_l1__(self):
        return NIRDALevel1HDUList(self)

    def plot_data(self, ax=None, **kwargs):
        ax_provided = ax is not None
        if ax is None:
            _, ax = plt.subplots()
        d = self["science"].data[0]
        k = d != 0
        vmin = kwargs.pop("vmin", np.nanpercentile(d[k], 1))
        vmax = kwargs.pop("vmax", np.nanpercentile(d[k], 1) + 100)
        im = ax.pcolormesh(
            self.column, self.row, d, vmin=vmin, vmax=vmax, **kwargs
        )
        ax.set(
            aspect="equal",
            xlabel="ROI Column",
            ylabel="ROI Row",
        )
        if not ax_provided:
            # Don't add titles for figures made for fits reports.
            ax.set(title=f"{self[0].header['targ_id']} {self.start_time.isot}")
        plt.colorbar(im, ax=ax)
        # ax.margins(0)
        return ax

    def describe(self):
        keys = [
            "TARG_ID",
            "TARG_RA",
            "TARG_DEC",
            "EXPOSRES",
            "RESETS1",
            "RESETS2",
            "DROPS1",
            "DROPS2",
            "DROPS3",
            "READS",
            "GRPS",
            "INTEGRTS",
            "ROISIZEX",
            "ROISIZEY",
            "TCLDTIP1",
        ]

        hdr = self[0].header
        rows = []
        for key in keys:
            try:
                value = hdr[key]
                comment = hdr.comments[key]
            except (KeyError, IndexError):
                value, comment = "N/A", ""
            if key in ("TARG_RA", "TARG_DEC", "TCLDTIP1"):
                try:
                    value = round(float(value), 4)
                except (TypeError, ValueError):
                    pass
            rows.append([key, value, comment])

        df = pd.DataFrame(rows, columns=["Key", "Value", "Comment"]).set_index(
            "Key"
        )
        return df

    def get_report_materials(self):
        report_materials = dict()
        report_materials["title"] = self[0].header.get("targ_id", "UNKNOWN")
        report_materials["subtitle"] = self.start_time.isot
        report_materials["report_metrics"] = self.describe()
        report_materials["report_plots"] = [
            lambda ax=None: self.plot_data(ax=ax)
        ]

        return report_materials


@register_hdulist(
    lambda h: (
        h
        and (
            (h[0].header.get("TELESCOP") == "NASA Pandora")
            & (h[0].header.get("INSTRMNT") == "NIRDA")
            & (h[0].header.get("PFCLASS") == "NIRDALevel1HDUList")
        )
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
    lambda h: (
        h
        and (
            (h[0].header.get("TELESCOP") == "NASA Pandora")
            & (h[0].header.get("INSTRMNT") == "NIRDA")
            & (h[0].header.get("PFCLASS") == "NIRDALevel2HDUList")
        )
    )
)
class NIRDALevel2HDUList(NIRDALevel1HDUList):
    filename = FORMATSDIR + "nirda/level2_nirda.xlsx"
    level = 2
