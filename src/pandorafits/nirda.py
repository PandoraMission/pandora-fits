# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.time import Time

from . import FORMATSDIR, NIRDAPRF, NIRDAReference, logger
from .fits import PandoraHDUList
from .io import register_hdulist

# from .scene import get_NIRDA_scene
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


def get_nirda_ramp_times(hdr):
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
class NIRDALevel0HDUList(PandoraHDUList):
    filename = FORMATSDIR + "nirda/level0_nirda.xlsx"
    reference = NIRDAReference
    prf = NIRDAPRF
    level = 0
    instrument = "NIRDA"

    def _update_data(self, data, time, exptime, integrations):
        """Helper function to update data when we sample"""
        self["SCIENCE"] = fits.CompImageHDU(data.value, name="SCIENCE")
        self["SCIENCE"].header["UNIT"] = data.unit.to_string()
        for hdu in ["TIME", "EXPTIME", "INTEGRATION"]:
            if hdu in self:
                self.pop(hdu)

        self.append(
            fits.ImageHDU(
                time.jd,
                name="TIME",
                header=fits.Header([("FRAME", "JD", "Time frame is JD")]),
            )
        )
        self.append(
            fits.ImageHDU(
                exptime.value,
                name="EXPTIME",
                header=fits.Header([("UNIT", "second", "Exposure time unit")]),
            )
        )
        self.append(
            fits.ImageHDU(
                integrations,
                name="INTEGRATION",
            )
        )

    def fix(self):
        if self.fixed:
            return self
        hdr = self[0].header
        acqd_ints = get_nirda_acqd_ints(hdr)
        # Fix the number of integrations and read frames so that we only keep whole integrations
        if acqd_ints != hdr["INTEGRTS"]:
            hdr["INTEGRTS"] = acqd_ints
            hdr["FRMSEXP"] = acqd_ints * hdr["GRPS"] * hdr["READS"]
            hdr["FRMSACQ"] = acqd_ints * hdr["GRPS"] * hdr["READS"]
            hdr["FRMSTOT"] = get_nirda_frame_indexes_tot(hdr)[
                acqd_ints * hdr["GRPS"] * hdr["READS"] - 1
            ]
            if hdr["GRPSAVGD"] == 0:
                idx = acqd_ints * hdr["GRPS"] * hdr["READS"]
            else:
                idx = acqd_ints * hdr["GRPS"]
            self = self.__class__(
                fits.HDUList(
                    [
                        fits.PrimaryHDU(header=hdr),
                        fits.CompImageHDU(
                            data=self[1].data[:idx], name="SCIENCE"
                        ),
                    ]
                )
            )
            logger.info("Fixed truncated NIRDA file.")

        hdr = self[0].header
        data = self[1].data
        shape = (
            hdr["INTEGRTS"],
            hdr["GRPS"],
            hdr["READS"],
            self[1].header["NAXIS2"],
            self[1].header["NAXIS1"],
        )
        t = get_nirda_frame_times(hdr).reshape(shape[:-2])
        exptime = get_nirda_ramp_times(hdr).reshape(shape[:-2]) * (
            hdr["FRMTIME"] * u.millisecond * hdr["READS"]
        ).to(u.second)
        I, _, _ = np.mgrid[: hdr["INTEGRTS"], : hdr["GRPS"], : hdr["READS"]]

        if hdr["GRPSAVGD"]:
            t = t[:, :, 0][:, :, None]
            exptime = exptime.mean(axis=2)[:, :, None]
            I = I[:, :, 0][:, :, None]
            data = data.reshape((shape[0], shape[1], 1, shape[3], shape[4]))
        else:
            data = data.reshape(shape)

        self._update_data((data * u.count), t, exptime, I)
        self[0].header["FIXED"] = True
        logger.info("Rearranged NIRDA file.")
        return self

    @property
    def fixed(self):
        hdr = self[0].header
        if "FIXED" in hdr:
            if hdr["FIXED"] == 1:
                return True
        return False

    @property
    def collapsed(self):
        hdr = self[0].header
        if "COLLAPS" in hdr:
            if hdr["COLLAPS"] == 1:
                return True
        return False

    def fowler_sample(self):
        if self.collapsed:
            return self
        if not self.fixed:
            self = self.fix()
        logger.info("Fowler sampling data.")
        hdr = self[0].header
        if hdr["GRPS"] > 1:
            t = self[2].data[:, :, 0]
            i = self[4].data[:, :, 0]
            if hdr["GRPSAVGD"] == 1:
                exptime = self[3].data[:, :, 0]
                d = self[1].data[:, :, 0].astype(float)
            elif hdr["GRPSAVGD"] == 0:
                exptime = self[3].data.mean(axis=2)
                d = self[1].data.astype(float).mean(axis=2)
        elif hdr["READS"] > 1:
            if hdr["GRPSAVGD"] == 1:
                # Can not fowler sample
                raise ValueError("No groups or reads to fowler sample.")
            elif hdr["GRPSAVGD"] == 0:
                i = self[4].data[:, 0, :]
                exptime = self[3].data[:, 0, :]
                t = self[2].data[:, 0, :]
                d = self[1].data.astype(float)[:, 0, :]

        self._update_data(
            ((d[:, -1] - d[:, 0]) * u.count)
            / (
                (exptime[:, -1] - exptime[:, 0])[:, None, None]
                * u.Quantity(1, self[3].header["unit"])
            ),
            Time(t[:, 1], format="jd"),
            (exptime[:, -1] - exptime[:, 0])
            * u.Quantity(1, self[3].header["unit"]),
            i[:, 1],
        )
        self[0].header["COLLAPS"] = 1
        logger.info("Fowlered sampled NIRDA file.")
        return self

    def difference_sample(self):
        if self.collapsed:
            return self
        if not self.fixed:
            self = self.fix()
        logger.info("Difference sampling data.")
        hdr = self[0].header
        shape = self[1].data.shape
        if (hdr["READS"] > 1) & (hdr["GRPSAVGD"] == 0):
            a = np.ones((shape[0], shape[1], shape[2] - 1), bool)
            d = np.diff(self[1].data.astype(float), axis=2)[a]
            t = self[2].data[:, :, 1:][a]
            exptime = np.diff(self[3].data, axis=2)[a]
            i = self[4].data[:, :, 1:][a]
        elif hdr["GRPS"] > 1:
            a = np.ones((shape[0], shape[1] - 1, shape[2]), bool)
            d = np.diff(self[1].data.astype(float), axis=1)[a]
            t = self[2].data[:, 1:, :][a]
            exptime = np.diff(self[3].data, axis=1)[a]
            i = self[4].data[:, 1:, :][a]
        else:
            raise ValueError(
                "Can not difference sample when integrations contain a single resultant read frame."
            )
        self._update_data(
            d
            * u.count
            / (exptime[:, None, None] * u.Quantity(1, self[3].header["unit"])),
            Time(t, format="jd"),
            exptime * u.Quantity(1, self[3].header["unit"]),
            i,
        )
        self[0].header["COLLAPS"] = 1
        logger.info("Difference sampled NIRDA file.")
        return self

    @property
    def time(self):
        return Time(self["TIME"].data, format="jd")

    @property
    def exptime(self):
        return u.Quantity(self["EXPTIME"].data, self["EXPTIME"].header["UNIT"])

    # @property
    # def time(self):
    #     hdr0 = self[0].header
    #     time = get_nirda_frame_times(hdr0).jd.reshape(
    #         (hdr0["INTEGRTS"], hdr0["GRPS"], hdr0["READS"])
    #     )[:, 0, 0]
    #     if self.fowlered:
    #         time = Time(
    #             (
    #                 time[:, None, None]
    #                 * np.ones((hdr0["INTEGRTS"], hdr0["GRPS"], hdr0["READS"]))
    #             )[:, 0, :].mean(axis=-1),
    #             format="jd",
    #         )
    #     else:
    #         if hdr0["GRPSAVGD"] == 0:
    #             time = Time(
    #                 (
    #                     time[:, None, None]
    #                     * np.ones((hdr0["INTEGRTS"], hdr0["GRPS"], hdr0["READS"]))
    #                 ).ravel(),
    #                 format="jd",
    #             )
    #         else:
    #             time = Time(
    #                 (time[:, None] * np.ones((hdr0["INTEGRTS"], hdr0["GRPS"]))).ravel(),
    #                 format="jd",
    #             )
    #     return time

    # @property
    # def integration_time(self):
    #     hdr0 = self[0].header
    #     ramp_time = get_nirda_ramp_times(hdr0).reshape(
    #         (hdr0["INTEGRTS"], hdr0["GRPS"], hdr0["READS"])
    #     ) * (hdr0["FRMTIME"] * u.millisecond).to(u.second)
    #     if self.fowlered:
    #         ramp_time = ramp_time[:, :, :].mean(axis=-1)
    #         ramp_time = ramp_time[:, -1] - ramp_time[:, 0]
    #     else:
    #         if hdr0["GRPSAVGD"] == 1:
    #             ramp_time = ramp_time[:, :, -1].ravel()
    #         else:
    #             ramp_time = ramp_time.ravel()
    #     return ramp_time

    def __to_l1__(self):
        return NIRDALevel1HDUList(self)

    def plot_data(self, ax=None, idx=0, **kwargs):
        if ax is None:
            _, ax = plt.subplots(dpi=150, facecolor="white")
        d = self["science"].data[idx]
        k = d != 0
        vmin = kwargs.pop("vmin", np.nanpercentile(d[k], 1))
        vmax = kwargs.pop("vmax", np.nanpercentile(d[k], 1) + 100)
        im = ax.pcolormesh(
            self.column, self.row, d, vmin=vmin, vmax=vmax, **kwargs
        )
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

    # def get_scene(self):
    #     hdr = self[0].header
    #     imcorner = (hdr["ROISTRTY"], hdr["ROISTRTX"])
    #     imshape = (hdr["ROISIZEY"], hdr["ROISIZEX"])
    #     return get_NIRDA_scene(
    #         time_jd=self.sequence_start_time.jd,
    #         ra=hdr["TARG_RA"],
    #         dec=hdr["TARG_DEC"],
    #         roll=hdr["TARG_RLL"],
    #         imcorner=imcorner,
    #         imshape=imshape,
    #     )

    # def _append_scene_extensions(self):
    #     hdr = self[0].header
    #     scene = self.get_scene()
    #     self.append(scene.get_catalog_hdu())
    #     self.append(scene.get_prf_hdu())
    #     self.append(scene.get_model_hdu())
    #     self.append(
    #         scene.get_aperture_hdu(
    #             SkyCoord(hdr["targ_ra"], hdr["targ_dec"], unit="deg"),
    #             relative_threshold=0.0001,
    #             absolute_threshold=30,
    #         )
    #     )
    #     self[0].header["GAIA_ID"] = self["APERTURE"].header["GAIA_ID"]
    #     logger.info("Appended scene extensions")

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
