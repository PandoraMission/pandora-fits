# Standard library
from datetime import timedelta

# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandoraspacecraft as psc
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from . import FORMATSDIR, VISDAReference, logger, ps
from .fits import PandoraHDUList
from .io import register_hdulist
from .report import ReportMixins
from .reshape import array_to_panels, panels_to_array, panels_to_cube
from .scene import get_VISDA_scene, get_VISDAFFI_scene
from .utils import convert_time

__all__ = [
    "VISDAFFILevel0HDUList",
    "VISDAFFILevel1HDUList",
    "VISDALevel0HDUList",
    "VISDALevel1HDUList",
    "VISDALevel2HDUList",
]


@register_hdulist(
    lambda h: (
        h
        and (
            (h[0].header.get("TELESCOP") == "NASA Pandora")
            & (h[0].header.get("INSTRMNT") == "VISDA")
            & ("FRMPCOAD" in h[0].header)
            & ("PFCLASS" not in h[0].header)
        )
    )
)
class VISDALevel0HDUList(ReportMixins, PandoraHDUList):
    filename = FORMATSDIR + "visda/level0_visda.xlsx"
    reference = VISDAReference
    level = 0
    instrument = "VISDA"

    def split(self, idxs=None):
        if idxs is None:
            idxs = np.arange(0, self.nROI)
        ndim = np.ndim(idxs)
        idxs = np.atleast_1d(idxs)
        # if self[0].header["NUMSTARS"] == 1:
        #     raise ValueError("Can not split, contains only one ROI.")
        pri = self[0].copy()
        pri.header["NUMSTARS"] = 1
        hdulists = []
        for tdx in idxs:
            im1 = fits.ImageHDU(
                self.data_list[:, tdx, :, :], self[1].header[10:]
            )
            tab1 = fits.TableHDU(self[2].data[[tdx]], self[2].header)
            hdulist = fits.HDUList([pri, im1, tab1, *self[3:]])
            hdulist = VISDALevel0HDUList(hdulist)
            hdulists.append(hdulist)
        if ndim == 0:
            return hdulists[0]
        return hdulists

    @property
    def border(self):
        numSubFrms = int(np.ceil(np.sqrt(self.nROI)))
        dims = (numSubFrms * self.ROI_size[0], numSubFrms * self.ROI_size[1])
        shape = (self[1].header["NAXIS2"], self[1].header["NAXIS1"])
        return shape[1] != dims[0]

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
            for x, y in Table(self["ROI_TABLE"].data).to_pandas().values
        ]

    @property
    def ROI_size(self):
        return (self[0].header["STARDIMS"], self[0].header["STARDIMS"])

    @property
    def nROI(self):
        return self[0].header["NUMSTARS"]

    @property
    def frmpcoad(self):
        return self[0].header["FRMPCOAD"]

    @property
    def data_cube(self):
        return panels_to_cube(
            self[1].data, nROI=self.nROI, ROI_size=self.ROI_size
        )

    @property
    def data_list(self):
        return panels_to_array(
            self[1].data, nROI=self.nROI, ROI_size=self.ROI_size
        )

    @property
    def list_row(self):
        row = np.asarray(
            [r + np.arange(self.ROI_size[0]) for r, _ in self.ROI_corners]
        )
        row = row[:, :, None] * np.ones((1, *self.ROI_size), dtype=int)
        return row

    @property
    def list_column(self):
        column = np.asarray(
            [c + np.arange(self.ROI_size[1]) for _, c in self.ROI_corners]
        )
        column = column[:, None, :] * np.ones((1, *self.ROI_size), dtype=int)
        return column

    @property
    def panel_row(self):
        R = array_to_panels(self.list_row.astype(float), border=self.border)
        R[self.border_mask] = np.nan
        return R

    @property
    def panel_column(self):
        R = array_to_panels(self.list_column.astype(float), border=self.border)
        R[self.border_mask] = np.nan
        return R

    @property
    def _central_target_index(self):
        return np.argmin(
            np.hypot(
                *(
                    (
                        np.asarray(self.ROI_corners)
                        + np.asarray(self.ROI_size)[0] / 2
                    )
                    - 1024
                ).T
            )
        )

    @property
    def border_mask(self):
        ex = int(self.border)
        num_stars = self.nROI
        num_sub_frames = int(np.ceil(np.sqrt(num_stars)))
        shape = self.ROI_size[0]
        mask = np.ones(
            (
                num_sub_frames * (shape + ex) + ex,
                num_sub_frames * (shape + ex) + ex,
            ),
            dtype=bool,
        )

        star = 0
        while star < num_stars:
            for i in range(num_sub_frames):
                yStrt = (num_sub_frames - (i + 1)) * (shape + ex) + ex
                for j in range(num_sub_frames):
                    xStrt = j * (shape + ex) + ex
                    mask[yStrt : yStrt + shape, xStrt : xStrt + shape] = False
                    star += 1
        return mask

    def plot_data(self, ax=None, **kwargs):
        called_from_report = ax is not None
        if ax is None:
            _, ax = plt.subplots()
        d = np.median(self["science"].data, axis=0)
        k = d != 0
        vmin = kwargs.pop("vmin", np.nanpercentile(d[k], 1))
        vmax = kwargs.pop("vmax", np.nanpercentile(d[k], 1) + 100)
        im = ax.pcolormesh(d, vmin=vmin, vmax=vmax, **kwargs)
        ax.set(
            aspect="equal",
            xlabel="Panel Column",
            ylabel="Panel Row",
        )
        if not called_from_report:
            ax.set(title=f"{self[0].header['targ_id']} {self.start_time.isot}")
        plt.colorbar(im, ax=ax)
        # ax.margins(0)
        return ax

    @property
    def astrometry(self):
        ra, dec, rot = np.asarray(
            [
                self["astrometry"].data[c]
                for c in ["RightAscension", "Declination", "Rotation"]
            ]
        )
        et = self[3].data["ExposureStartTime_us"]
        et = et[: len(ra)]
        if et[-1] > 2**60:
            et = et[:-1]
            ra, dec, rot = ra[:-1], dec[:-1], rot[:-1]

        et = (
            convert_time(
                self[0].header["CORSTIME"], self[0].header["FINETIME"]
            )
            + et * u.ms
        )
        return et, ra, dec, rot

    @property
    def hot_pixels(self):
        # Count pixels per coadded frame that exceed median + 5 × σ_MAD.
        #
        # MAD (Median Absolute Deviation) scaled by 1.4826 gives a Gaussian-
        # equivalent σ that is robust to the outliers being detected:
        # a handful of hot or cosmic-ray pixels inflates the standard deviation
        # dramatically but moves the median by less than one count, so the
        # threshold stays anchored to the bulk of the pixel distribution
        # (Rousseeuw & Croux 1993).  5 σ_MAD gives a false-alarm rate of
        # ~6 × 10⁻⁷ per pixel per frame for Gaussian noise — aggressive enough
        # to catch real defects while avoiding flagging faint stars.
        #
        # Border pixels (structural zeros between sub-frame panels) are excluded
        # via border_mask so they do not bias the median or inflate the count.
        science = self["science"].data  # (nframes, panel_h, panel_w)
        valid = ~self.border_mask.ravel()
        pixels = science.reshape(self.nframes, -1).astype(float)[:, valid]
        med = np.median(pixels, axis=1, keepdims=True)
        sigma_mad = 1.4826 * np.median(np.abs(pixels - med), axis=1, keepdims=True)
        counts = (pixels > med + 5.0 * sigma_mad).sum(axis=1)
        return self.time, counts

    @property
    def dead_pixels(self):
        # Count pixels reading zero DN per coadded frame within valid ROI regions.
        #
        # A pixel that accumulates zero counts across every sub-exposure in a coadd
        # has failed to register charge, the standard indicator of a dead or
        # permanently trapped pixel in a solid-state detector (Janesick 2001).
        # Zero is the correct floor because coadding sums sub-exposures, so any
        # genuine signal grows proportionally while a non-responsive pixel remains
        # stuck at zero regardless of scene brightness. Inter-panel border pixels,
        # which are also zero by construction, are excluded via border_mask to avoid
        # conflating structural padding with detector defects.
        science = self["science"].data  # (nframes, panel_h, panel_w)
        valid = ~self.border_mask.ravel()
        counts = (science.reshape(self.nframes, -1)[:, valid] == 0).sum(axis=1)
        return self.time, counts

    def get_position_data(self):
        et, ra, dec, rot = self.astrometry
        dt = timedelta(seconds=self.frame_time.to(u.second).value)
        t = self.start_time
        df = []
        count, missing = [], []
        for frame in np.arange(self.nframes):
            j = (et >= (t + (dt * frame))) & (et < (t + (dt * (frame + 1))))
            k = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(rot)
            k &= (ra != 0) & (dec != 0) & (rot != 0)
            count.append(len(k[j]))
            missing.append((~k[j]).sum())
            k &= j
            if not k.any():
                df.append(
                    pd.DataFrame(
                        np.asarray([np.nan] * 6)[None, :],
                        columns=[
                            "avg_ra",
                            "avg_dec",
                            "avg_rot",
                            "err_ra",
                            "err_dec",
                            "err_rot",
                        ],
                    )
                )
                continue
            avg_ra, avg_dec, avg_rot = (
                np.mean(ra[k]),
                np.mean(dec[k]),
                np.mean(rot[k]),
            )
            err_ra, err_dec, err_rot = (
                np.median(np.abs(ra[k] - avg_ra)),
                np.median(np.abs(dec[k] - avg_dec)),
                np.median(np.abs(rot[k] - avg_rot)),
            )
            df.append(
                pd.DataFrame(
                    np.asarray(
                        [avg_ra, avg_dec, avg_rot, err_ra, err_dec, err_rot]
                    )[None, :],
                    columns=[
                        "avg_ra",
                        "avg_dec",
                        "avg_rot",
                        "err_ra",
                        "err_dec",
                        "err_rot",
                    ],
                )
            )
        df = pd.concat(df).reset_index(drop=True)
        t = self.time.jd
        df["t"] = t
        df["sep"] = np.hypot(
            df.avg_ra.values - self.targ.ra.value,
            df.avg_dec.values - self.targ.dec.value,
        )
        df["count"] = count
        df["missing"] = missing
        if len(df) <= 3:
            df["stability"] = np.nan
            df["recall"] = np.nan
            df["stable"] = False
        else:
            df["stability"] = (
                np.hypot(
                    # np.gradient(df.err_ra.values, t), np.gradient(df.err_dec.values, t)
                    df.err_ra.values,
                    df.err_dec.values,
                )
                * 3600
            )
            df.loc[df["count"] < 5, "stability"] = np.nan
            df["recall"] = (
                np.hypot(
                    np.gradient(
                        df.avg_ra.values - np.nanmedian(df.avg_ra.values), t
                    ),
                    np.gradient(
                        df.avg_dec.values - np.nanmedian(df.avg_dec.values), t
                    ),
                )
                * 3600
                / 86400
            )
        return df

    def plot_astrometry(self, ax=None, **kwargs):
        called_from_report = ax is not None
        if ax is None:
            _, ax = plt.subplots()
        df = self.get_position_data()
        t_min = (df.t.values - self.start_time.jd) * 24 * 60
        ax.plot(t_min, (df.avg_ra.values - df.avg_ra.mean()) * 3600, label="RA")
        ax.plot(t_min, (df.avg_dec.values - df.avg_dec.mean()) * 3600, label="Dec")
        ax.legend()
        ax.set(
            xlabel="Time from Start [min]",
            ylabel="Position - Mean Position\n[arcsecond]",
        )
        if not called_from_report:
            ax.set(title=f"{self[0].header['targ_id']} {self.start_time.isot}")
        return ax

    def plot_bad_pixels(self, ax=None, **kwargs):
        called_from_report = ax is not None
        if ax is None:
            _, ax = plt.subplots()
        t, hot = self.hot_pixels
        _, dead = self.dead_pixels
        t_min = (t.jd - self.start_time.jd) * 24 * 60
        ax.plot(t_min, hot, label="Hot", **kwargs)
        if np.any(dead > 0):
            ax.plot(t_min, dead, label="Dead", **kwargs)
        ax.legend()
        ax.set(
            yscale='linear',
            xlabel="Time from Start [min]",
            ylabel="Pixel Count",
        )
        if not called_from_report:
            ax.set(title=f"{self[0].header['targ_id']} {self.start_time.isot}")
        return ax

    def describe(self):
        keys = [
            "NUMSTARS",
            "TARG_ID",
            "TARG_RA",
            "TARG_DEC",
            "FRMSREQD",
            "FRMSCLCT",
            "STARDIMS",
            "NUMPCOAD",
            "FRMPCOAD",
            "TARG_RLL",
        ]

        hdr = self[0].header
        rows = []
        for key in keys:
            try:
                value = hdr[key]
                comment = hdr.comments[key]
            except (KeyError, IndexError):
                value, comment = "N/A", ""
            if key in ("TARG_RA", "TARG_DEC"):
                try:
                    value = round(float(value), 4)
                except (TypeError, ValueError):
                    pass
            rows.append([key, value, comment])

        _, hot = self.hot_pixels
        _, dead = self.dead_pixels
        rows.append(["HOT_MED", int(np.median(hot)), "Median hot pixels per frame"])
        rows.append(["DEAD_MED", int(np.median(dead)), "Median dead pixels per frame"])

        return pd.DataFrame(
            rows, columns=["Key", "Value", "Comment"]
        ).set_index("Key")

    def get_report_materials(self):
        return {
            "title": self[0].header["targ_id"],
            "subtitle": self.start_time.isot,
            "tables": self.describe(),
            "star_field": lambda ax=None: self.plot_data(ax=ax),
            "astrometry": lambda ax=None: self.plot_astrometry(ax=ax),
            "bad_pixels": lambda ax=None: self.plot_bad_pixels(ax=ax),
        }

    def get_earth_angle(self):
        return ps.get_angle_to_body(
            self.time,
            direction="z",
            body="earth",
            pointing_vecs=psc.utils.radec_to_vec(self.targ_ra, self.targ_dec),
        )

    def __to_l1__(self):
        return VISDALevel1HDUList(self)


@register_hdulist(
    lambda h: (
        h
        and (
            (h[0].header.get("TELESCOP") == "NASA Pandora")
            & (h[0].header.get("INSTRMNT") == "VISDA")
            & (h[0].header.get("PFCLASS") == "VISDALevel1HDUList")
        )
    )
)
class VISDALevel1HDUList(VISDALevel0HDUList):
    filename = FORMATSDIR + "visda/level1_visda.xlsx"
    level = 1

    def get_scene(self):
        hdr = self[0].header
        return get_VISDA_scene(
            time_jd=self.sequence_start_time.jd,
            ra=hdr["TARG_RA"],
            dec=hdr["TARG_DEC"],
            roll=hdr["TARG_RLL"],
            ROI_corners=tuple(self.ROI_corners),
            ROI_size=self.ROI_size,
        )

    def _get_aperture_and_catalog(self, scene):
        cataloghdu = scene.get_catalog_hdu()
        df = Table(cataloghdu.data).to_pandas()
        (
            aper,
            df["contamination"],
            df["completeness"],
            df["total_in_aperture"],
        ) = scene.get_all_apertures()
        hdr = fits.Header(
            [
                fits.Card(*c)
                for c in [
                    (
                        "IMSIZE0",
                        scene.prf.imshape[0],
                        "Size of the full detector image in ROW",
                    ),
                    (
                        "IMCRNR0",
                        scene.prf.imcorner[0],
                        "Corner of the image in ROW.",
                    ),
                    (
                        "IMSIZE1",
                        scene.prf.imshape[1],
                        "Size of the full detector image in COLUMN",
                    ),
                    (
                        "IMCRNR1",
                        scene.prf.imcorner[1],
                        "Corner of the image in COLUMN.",
                    ),
                ]
            ]
        )
        aperturehdu = fits.CompImageHDU(
            data=array_to_panels(
                aper if aper.ndim == 4 else aper[:, None, :, :],
                border=self.border,
            ).astype(np.int16),
            name="APERTURE",
            header=hdr,
        )
        cataloghdu = fits.convenience.table_to_hdu(Table.from_pandas(df))
        cataloghdu.header["EXTNAME"] = "CATALOG"
        return aperturehdu, cataloghdu

    def _append_scene_extensions(self):
        scene = self.get_scene()
        self.append(fits.ImageHDU(self.panel_row, name="PIXEL_ROW"))
        self.append(fits.ImageHDU(self.panel_column, name="PIXEL_COLUMN"))
        aperturehdu, cataloghdu = self._get_aperture_and_catalog(scene)
        self.append(cataloghdu)
        self.append(scene.get_prf_hdu())
        modelhdu = scene.get_model_hdu()
        self.append(
            fits.ImageHDU(
                array_to_panels(
                    (
                        modelhdu.data
                        if modelhdu.data.ndim == 3
                        else modelhdu.data[None, :, :]
                    ),
                    border=self.border,
                ),
                header=modelhdu.header[8:],
                name="MODEL_IMAGE",
            )
        )
        self.append(aperturehdu)
        logger.info("Appended scene extensions")

    def __to_l2__(self):
        return VISDALevel2HDUList(self)


@register_hdulist(
    lambda h: (
        h
        and (
            (h[0].header.get("TELESCOP") == "NASA Pandora")
            & (h[0].header.get("INSTRMNT") == "VISDA")
            & (h[0].header.get("PFCLASS") == "VISDALevel2HDUList")
        )
    )
)
class VISDALevel2HDUList(VISDALevel1HDUList):
    filename = FORMATSDIR + "visda/level2_visda.xlsx"
    level = 2


@register_hdulist(
    lambda h: (
        h
        and (
            (h[0].header.get("TELESCOP") == "NASA Pandora")
            & (h[0].header.get("INSTRMNT") == "VISDA")
            & ("FRMPCOAD" not in h[0].header)
            & ("PFCLASS" not in h[0].header)
        )
    )
)
class VISDAFFILevel0HDUList(ReportMixins, PandoraHDUList):
    filename = FORMATSDIR + "visda/level0-ffi_visda.xlsx"
    reference = VISDAReference
    level = 0
    instrument = "VISDA"

    def plot_data(self, ax=None, **kwargs):
        called_from_report = ax is not None
        if ax is None:
            _, ax = plt.subplots()
        d = self["science"].data[0]
        k = d != 0
        vmin = kwargs.pop("vmin", np.nanpercentile(d[k], 1))
        vmax = kwargs.pop("vmax", np.nanpercentile(d[k], 1) + 100)
        im = ax.pcolormesh(d, vmin=vmin, vmax=vmax, **kwargs)
        ax.set(
            aspect="equal",
            xlabel="Column",
            ylabel="Row",
        )
        if not called_from_report:
            ax.set(title=f"{self[0].header['targ_id']} {self.start_time.isot}")
        plt.colorbar(im, ax=ax)
        # ax.margins(0)
        return ax

    def describe(self):
        keys = [
            "TARG_ID",
        ]

        hdr = self[0].header
        df = pd.DataFrame(
            np.asarray([hdr.cards[key] for key in keys]),
            columns=["Key", "Value", "Comment"],
        ).set_index("Key")
        return df

    def get_report_materials(self):
        return [
            lambda ax=None: self.plot_data(ax=ax),
            None,
            None,
            lambda ax=None: self.plot_description(ax=ax),
        ]

    def __to_l1__(self):
        return VISDAFFILevel1HDUList(self)


@register_hdulist(
    lambda h: (
        h
        and (
            (h[0].header.get("TELESCOP") == "NASA Pandora")
            & (h[0].header.get("INSTRMNT") == "VISDA")
            & (h[0].header.get("PFCLASS") == "VISDAFFILevel1HDUList")
        )
    )
)
class VISDAFFILevel1HDUList(VISDAFFILevel0HDUList):
    filename = FORMATSDIR + "visda/level1-ffi_visda.xlsx"
    level = 1

    def get_scene(self):
        hdr = self[0].header
        imcorner = (hdr["ROISTRTY"], hdr["ROISTRTX"])
        imshape = (hdr["ROISIZEY"], hdr["ROISIZEX"])
        return get_VISDAFFI_scene(
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
        self.append(
            scene.get_aperture_hdu(
                SkyCoord(hdr["targ_ra"], hdr["targ_dec"], unit="deg"),
                relative_threshold=0.005,
                absolute_threshold=50,
            )
        )
        self[0].header["GAIA_ID"] = self["APERTURE"].header["GAIA_ID"]
        logger.info("Appended scene extensions")

    def __to_l2__(self):
        return VISDAFFILevel2HDUList(self)


@register_hdulist(
    lambda h: (
        h
        and (
            (h[0].header.get("TELESCOP") == "NASA Pandora")
            & (h[0].header.get("INSTRMNT") == "VISDA")
            & (h[0].header.get("PFCLASS") == "VISDAFFILevel2HDUList")
        )
    )
)
class VISDAFFILevel2HDUList(VISDAFFILevel1HDUList):
    filename = FORMATSDIR + "visda/level2-ffi_visda.xlsx"
    level = 2
