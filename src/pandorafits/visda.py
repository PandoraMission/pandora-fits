import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from . import FORMATSDIR, VISDAReference, logger
from .fits import PandoraHDUList
from .io import register_hdulist
from .report import ReportMixins
from .reshape import list_to_panels, panels_to_cube, panels_to_list
from .scene import get_VISDA_scene, get_VISDAFFI_scene

__all__ = [
    "VISDAFFILevel0HDUList",
    "VISDAFFILevel1HDUList",
    "VISDALevel0HDUList",
    "VISDALevel1HDUList",
    "VISDALevel2HDUList",
]


@register_hdulist(
    lambda h: h
    and (
        (h[0].header.get("TELESCOP") == "NASA Pandora")
        & (h[0].header.get("INSTRMNT") == "VISDA")
        & ("FRMPCOAD" in h[0].header)
        & ("PFCLASS" not in h[0].header)
    )
)
class VISDALevel0HDUList(ReportMixins, PandoraHDUList):
    filename = FORMATSDIR + "visda/level0_visda.xlsx"
    reference = VISDAReference
    level = 0
    instrument = "VISDA"

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
    def data_cube(self):
        return panels_to_cube(self[1].data, nROI=self.nROI, ROI_size=self.ROI_size)

    @property
    def data_list(self):
        return panels_to_list(self[1].data, nROI=self.nROI, ROI_size=self.ROI_size)

    @property
    def list_row(self):
        row = np.asarray([r + np.arange(self.ROI_size[0]) for r, _ in self.ROI_corners])
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
        R = list_to_panels(self.list_row.astype(float), border=self.border)
        R[self.border_mask] = np.nan
        return R

    @property
    def panel_column(self):
        R = list_to_panels(self.list_column.astype(float), border=self.border)
        R[self.border_mask] = np.nan
        return R

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
            xlabel="Panel Column",
            ylabel="Panel Row",
        )
        plt.colorbar(im, ax=ax)
        # ax.margins(0)
        return ax

    def plot_astrometry(self, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots()
        t = Table(self["temp_time"].data).to_pandas()
        d = Table(self["astrometry"].data).to_pandas()
        ax.plot(
            t.ExposureStartTime_us.values / 1e3,
            (d.RightAscension.values - d.RightAscension.mean()) * 3600,
            label="RA",
        )
        ax.plot(
            t.ExposureStartTime_us.values / 1e3,
            (d.Declination.values - d.Declination.mean()) * 3600,
            label="Dec",
        )
        ax.legend()
        ax.set(
            title=f"{self[0].header['targ_id']} {self.start_time.isot}",
            xlabel="Time in Exposure [s]",
            ylabel="Position - Mean Position [arcsecond]",
        )
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
    lambda h: h
    and (
        (h[0].header.get("TELESCOP") == "NASA Pandora")
        & (h[0].header.get("INSTRMNT") == "VISDA")
        & (h[0].header.get("PFCLASS") == "VISDALevel1HDUList")
    )
)
class VISDALevel1HDUList(VISDALevel0HDUList):
    filename = FORMATSDIR + "visda/level1_visda.xlsx"
    level = 1

    def split(self):
        if self[0].header["NUMSTARS"] == 1:
            raise ValueError("Can not split, contains only one ROI.")
        pri = self[0].copy()
        pri.header["NUMSTARS"] = 1
        hdulists = []
        for tdx in range(self.nROI):
            im1 = fits.ImageHDU(self.data_list[:, tdx, :, :], self[1].header[10:])
            tab1 = fits.TableHDU(self[2].data[[tdx]], self[2].header)
            hdulist = fits.HDUList([pri, im1, tab1, self[3], self[4]])
            hdulist = VISDALevel1HDUList(hdulist)
            hdulists.append(hdulist)
        return hdulists

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
        aper, df["contamination"], df["completeness"], df["total_in_aperture"] = (
            scene.get_all_apertures()
        )
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
            data=list_to_panels(
                aper if aper.ndim == 4 else aper[:, None, :, :], border=self.border
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
                list_to_panels(
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
    lambda h: h
    and (
        (h[0].header.get("TELESCOP") == "NASA Pandora")
        & (h[0].header.get("INSTRMNT") == "VISDA")
        & (h[0].header.get("PFCLASS") == "VISDALevel2HDUList")
    )
)
class VISDALevel2HDUList(VISDALevel1HDUList):
    filename = FORMATSDIR + "visda/level2_visda.xlsx"
    level = 2


@register_hdulist(
    lambda h: h
    and (
        (h[0].header.get("TELESCOP") == "NASA Pandora")
        & (h[0].header.get("INSTRMNT") == "VISDA")
        & ("FRMPCOAD" not in h[0].header)
        & ("PFCLASS" not in h[0].header)
    )
)
class VISDAFFILevel0HDUList(ReportMixins, PandoraHDUList):
    filename = FORMATSDIR + "visda/level0-ffi_visda.xlsx"
    reference = VISDAReference
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
    lambda h: h
    and (
        (h[0].header.get("TELESCOP") == "NASA Pandora")
        & (h[0].header.get("INSTRMNT") == "VISDA")
        & (h[0].header.get("PFCLASS") == "VISDAFFILevel1HDUList")
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
    lambda h: h
    and (
        (h[0].header.get("TELESCOP") == "NASA Pandora")
        & (h[0].header.get("INSTRMNT") == "VISDA")
        & (h[0].header.get("PFCLASS") == "VISDAFFILevel2HDUList")
    )
)
class VISDAFFILevel2HDUList(VISDAFFILevel1HDUList):
    filename = FORMATSDIR + "visda/level2-ffi_visda.xlsx"
    level = 2
