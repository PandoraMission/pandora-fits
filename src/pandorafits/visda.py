# Standard library
from datetime import timedelta

# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

from . import FORMATSDIR, VISDAPRF, VISDAReference, logger
from .fits import PandoraHDUList
from .io import register_hdulist
from .report import ReportMixins
from .reshape import array_to_panels, panels_to_array, panels_to_cube

# from .scene import get_VISDA_scene, get_VISDAFFI_scene

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
    prf = VISDAPRF
    level = 0
    instrument = "VISDA"

    def fix(self):
        for hdu in ["TIME", "EXPTIME"]:
            if hdu in self:
                self.pop(hdu)

        self.append(
            fits.ImageHDU(
                self.time.jd,
                name="TIME",
                header=fits.Header([("FRAME", "JD", "Time frame is JD")]),
            )
        )
        self.append(
            fits.ImageHDU(
                self.exptime.value,
                name="EXPTIME",
                header=fits.Header([("UNIT", "second", "Exposure time unit")]),
            )
        )
        self[0].header["FIXED"] = True
        logger.info("Rearranged VISDA file.")
        return self

    @property
    def nframes(self):
        return self[0].header["NUMPCOAD"]

    @property
    def ncoadds(self):
        return self[0].header["FRMPCOAD"]

    @property
    def frame_time(self):
        return self.read_time * self.ncoadds

    @property
    def time(self):
        if "TIME" not in self:
            dt = timedelta(seconds=self.frame_time.to(u.second).value)
            return self.start_time + (np.arange(self.nframes) * dt)
        else:
            return Time(self["TIME"].data, format="jd")

    @property
    def exptime(self):
        if "EXPTIME" not in self:
            return np.ones(self.nframes) * self.frame_time.to(u.second)
        else:
            return u.Quantity(
                self["EXPTIME"].data, unit=self["EXPTIME"].header["UNIT"]
            )

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
                self.data_list[:, tdx, :, :], header=self[1].header[10:]
            )
            tab1 = fits.TableHDU(self[2].data[[tdx]], header=self[2].header)
            hdulist = fits.HDUList([pri, im1, tab1, *self[3:]])
            hdulist = self.__class__(hdulist)
            hdulists.append(hdulist)
        if ndim == 0:
            return hdulists[0]
        return hdulists

    def split_target(self):
        return self.split(self._central_target_index)

    @property
    def border(self):
        numSubFrms = int(np.ceil(np.sqrt(self.nROI)))
        dims = (numSubFrms * self.ROI_size[0], numSubFrms * self.ROI_size[1])
        shape = (self[1].header["NAXIS2"], self[1].header["NAXIS1"])
        return shape[1] != dims[0]

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
        if "ROI_TABLE" in self:
            R = array_to_panels(
                self.list_row.astype(float), border=self.border
            )
            R[self.border_mask] = np.nan
            return R
        return np.arange(
            self[0].header["ROISTRTY"],
            self[0].header["ROISTRTY"] + self[0].header["ROISIZEY"],
        )[:, None] * np.ones(self[0].header["ROISIZEX"], dtype=int)

    @property
    def panel_column(self):
        if "ROI_TABLE" in self:
            C = array_to_panels(
                self.list_column.astype(float), border=self.border
            )
            C[self.border_mask] = np.nan
            return C
        return (
            np.arange(
                self[0].header["ROISTRTX"],
                self[0].header["ROISTRTX"] + self[0].header["ROISIZEX"],
            )[None, :]
            * np.ones(self[0].header["ROISIZEY"], dtype=int)[:, None]
        )

    @property
    def _central_target_index(self):
        if self.wcs is not None:
            x, y = self.wcs.world_to_pixel(self.coord)
        else:
            x, y = 1024, 1024
        idx = np.argmin(
            np.hypot(
                np.asarray(self.ROI_corners)[:, 0] + self.ROI_size[0] / 2 - y,
                np.asarray(self.ROI_corners)[:, 1] + self.ROI_size[0] / 2 - x,
            )
        )
        if self.wcs is not None:
            yc, xc = self.ROI_corners[idx]
            if not (
                (x > (xc - 5))
                & (x < (xc + self.ROI_size[0] + 5))
                & (y > (yc - 5))
                & (y < (yc + self.ROI_size[1] + 5))
            ):
                raise ValueError("Can not find target ROI.")
        return idx

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

    def plot_data(self, ax=None, idx=0, **kwargs):
        if ax is None:
            _, ax = plt.subplots(dpi=250, facecolor="white")
        d = self["science"].data[idx]
        k = d != 0
        vmin = kwargs.pop("vmin", np.nanpercentile(d[k], 1))
        vmax = kwargs.pop("vmax", np.nanpercentile(d[k], 1) + 100)
        im = ax.pcolormesh(d, vmin=vmin, vmax=vmax, **kwargs)
        ax.set(
            aspect="equal",
            title=f"{self[0].header['targ_id']} {self.start_time.isot}",
            xlabel="Panel Column",
            ylabel="Panel Row",
        )
        plt.colorbar(im, ax=ax)
        # ax.margins(0)
        return ax

    def describe(self):
        keys = [
            "NUMSTARS",
            "TARG_ID",
            "TARG_RA",
            "TARG_DEC",
            "FRMSREQD",
            "FRMSCLCT",
            "NUMSTARS",
            "STARDIMS",
            "NUMPCOAD",
            "FRMPCOAD",
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
            lambda ax=None: self.plot_astrometry(ax=ax),
            None,
            lambda ax=None: self.plot_description(ax=ax),
        ]

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

    # def get_scene(self):
    #     hdr = self[0].header
    #     return get_VISDA_scene(
    #         time_jd=self.sequence_start_time.jd,
    #         ra=hdr["TARG_RA"],
    #         dec=hdr["TARG_DEC"],
    #         roll=hdr["TARG_RLL"],
    #         ROI_corners=tuple(self.ROI_corners),
    #         ROI_size=self.ROI_size,
    #     )

    # def _get_aperture_and_catalog(self, scene):
    #     cataloghdu = scene.get_catalog_hdu()
    #     df = Table(cataloghdu.data).to_pandas()
    #     (
    #         aper,
    #         df["contamination"],
    #         df["completeness"],
    #         df["total_in_aperture"],
    #     ) = scene.get_all_apertures()
    #     hdr = fits.Header(
    #         [
    #             fits.Card(*c)
    #             for c in [
    #                 (
    #                     "IMSIZE0",
    #                     scene.prf.imshape[0],
    #                     "Size of the full detector image in ROW",
    #                 ),
    #                 (
    #                     "IMCRNR0",
    #                     scene.prf.imcorner[0],
    #                     "Corner of the image in ROW.",
    #                 ),
    #                 (
    #                     "IMSIZE1",
    #                     scene.prf.imshape[1],
    #                     "Size of the full detector image in COLUMN",
    #                 ),
    #                 (
    #                     "IMCRNR1",
    #                     scene.prf.imcorner[1],
    #                     "Corner of the image in COLUMN.",
    #                 ),
    #             ]
    #         ]
    #     )
    #     aperturehdu = fits.CompImageHDU(
    #         data=array_to_panels(
    #             aper if aper.ndim == 4 else aper[:, None, :, :],
    #             border=self.border,
    #         ).astype(np.int16),
    #         name="APERTURE",
    #         header=hdr,
    #     )
    #     cataloghdu = fits.convenience.table_to_hdu(Table.from_pandas(df))
    #     cataloghdu.header["EXTNAME"] = "CATALOG"
    #     return aperturehdu, cataloghdu

    # def _append_scene_extensions(self):
    #     scene = self.get_scene()
    #     self.append(fits.ImageHDU(self.panel_row, name="PIXEL_ROW"))
    #     self.append(fits.ImageHDU(self.panel_column, name="PIXEL_COLUMN"))
    #     aperturehdu, cataloghdu = self._get_aperture_and_catalog(scene)
    #     self.append(cataloghdu)
    #     self.append(scene.get_prf_hdu())
    #     modelhdu = scene.get_model_hdu()
    #     self.append(
    #         fits.ImageHDU(
    #             array_to_panels(
    #                 (
    #                     modelhdu.data
    #                     if modelhdu.data.ndim == 3
    #                     else modelhdu.data[None, :, :]
    #                 ),
    #                 border=self.border,
    #             ),
    #             header=modelhdu.header[8:],
    #             name="MODEL_IMAGE",
    #         )
    #     )
    #     self.append(aperturehdu)
    #     logger.info("Appended scene extensions")

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
    prf = VISDAPRF
    level = 0
    instrument = "VISDA"

    def plot_data(self, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots()
        d = self["science"].data[0]
        k = d != 0
        vmin = kwargs.pop("vmin", np.nanpercentile(d[k], 1))
        vmax = kwargs.pop("vmax", np.nanpercentile(d[k], 1) + 100)
        im = ax.pcolormesh(d, vmin=vmin, vmax=vmax, **kwargs)
        ax.set(
            aspect="equal",
            title=f"{self[0].header['targ_id']} {self.start_time.isot}",
            xlabel="Column",
            ylabel="Row",
        )
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

    # def get_scene(self):
    #     hdr = self[0].header
    #     imcorner = (hdr["ROISTRTY"], hdr["ROISTRTX"])
    #     imshape = (hdr["ROISIZEY"], hdr["ROISIZEX"])
    #     return get_VISDAFFI_scene(
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
    #     self.append(
    #         scene.get_aperture_hdu(
    #             SkyCoord(hdr["targ_ra"], hdr["targ_dec"], unit="deg"),
    #             relative_threshold=0.005,
    #             absolute_threshold=50,
    #         )
    #     )
    #     self[0].header["GAIA_ID"] = self["APERTURE"].header["GAIA_ID"]
    #     logger.info("Appended scene extensions")

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
