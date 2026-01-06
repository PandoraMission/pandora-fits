import numpy as np
import pandoraaperture as pa
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from . import FORMATSDIR, VISDAReference, logger
from .fits import PandoraHDUList
from .reshape import list_to_panels, panels_to_cube, panels_to_list

__all__ = [
    "VISDAFFILevel0HDUList",
    "VISDAFFILevel1HDUList",
    "VISDALevel0HDUList",
    "VISDALevel1HDUList",
    "VISDALevel2HDUList",
]


class VISDALevel0HDUList(PandoraHDUList):
    filename = FORMATSDIR + "visda/level0_visda.xlsx"
    reference = VISDAReference

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
        return panels_to_cube(self[1].data, nROI=self.nROI, ROI_size=self.ROI_size)

    #     numSubFrms = int(np.ceil(np.sqrt(self.nROI)))
    #     dims = (numSubFrms * self.ROI_size[0], numSubFrms * self.ROI_size[1])
    #     if self[1].header["NAXIS1"] == dims[0]:
    #         ex = 0
    #     elif self[1].header["NAXIS1"] == (dims[0] + numSubFrms + 1):
    #         ex = 1
    #     else:
    #         raise ValueError("Can not parse the padding dimensions.")
    #     data = self[1].data[:, ex:, ex:]
    #     shape = data.shape[1:]
    #     width = self.ROI_size[0]
    #     nims = int(shape[1] / (width + ex))
    #     d1 = np.asarray(np.array_split(data, nims, axis=2))
    #     nims = int(shape[0] / (width + ex))
    #     d2 = np.asarray(np.array_split(d1, nims, axis=2))[
    #         :, :, :, :-ex, :-ex
    #     ].transpose([2, 0, 1, 3, 4])
    #     return d2

    @property
    def data_list(self):
        return panels_to_list(self[1].data, nROI=self.nROI, ROI_size=self.ROI_size)
        # d = self.data_cube[:, ::-1]
        # starlist = []
        # numSubFrms = int(np.ceil(np.sqrt(self.nROI)))
        # for idx in range(numSubFrms):
        #     for jdx in range(numSubFrms):data.sha
        #         if (idx * numSubFrms + jdx) == self.nROI:
        #             break
        #         starlist.append(d[:, idx, jdx])
        # return np.asarray(starlist, dtype=starlist[0].dtype).transpose([1, 0, 2, 3])

    def data_list_to_panels(self, data_list):
        return list_to_panels(data_list, border=self.border)
        # numSubFrms = int(np.ceil(np.sqrt(self.nROI)))
        # dims = (numSubFrms * self.ROI_size[0], numSubFrms * self.ROI_size[1])
        # if self[1].header["NAXIS1"] == dims[0]:
        #     ex = 0
        # elif self[1].header["NAXIS1"] == (dims[0] + numSubFrms + 1):
        #     ex = 1
        # else:
        #     raise ValueError("Can not parse the padding dimensions.")
        # width = self.ROI_size[0]
        # data = np.zeros(
        #     (
        #         self.nframes,
        #         (width + ex) * numSubFrms + ex,
        #         (width + ex) * numSubFrms + ex,
        #     ),
        #     dtype=star_list.dtype,
        # )
        # for idx in range(numSubFrms):
        #     for jdx in range(numSubFrms):
        #         if (idx * numSubFrms + jdx) == self.nROI:
        #             break
        #         kdx = numSubFrms - 1 - idx
        #         data[
        #             :,
        #             (ex + idx * ex) + (idx * width) : (ex + idx * ex)
        #             + ((idx + 1) * width),
        #             (ex + jdx * ex) + (jdx * width) : (ex + jdx * ex)
        #             + ((jdx + 1) * width),
        #         ] = star_list[:, kdx * numSubFrms + jdx]
        # return data

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

    def __to_l1__(self):
        return VISDALevel1HDUList(self)


class VISDALevel1HDUList(VISDALevel0HDUList):
    filename = FORMATSDIR + "visda/level1_visda.xlsx"
    reference = VISDAReference

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
        if self[0].header["NUMSTARS"] == 1:
            prf = pa.SpatialPRF.from_reference().to_PRF(
                np.asarray(self.ROI_corners[0]) + np.asarray(self.ROI_size) / 2
            )
            prf.imcorner = self.ROI_corners[0]
            prf.imshape = self.ROI_size
            scene = pa.SkyScene(prf, self.wcs, self.start_time)
        else:
            prf = pa.SpatialPRF.from_reference()
            prf.imcorner = (0, 0)
            prf.imshape = (2048, 2048)
            corners = [
                tuple(i) for i in np.asarray(self.ROI_corners)
            ]  # + np.asarray((hdr['ROISTRTX'], hdr['ROISTRTY']))]
            scene = pa.ROISkyScene(
                prf,
                self.wcs,
                time=self.start_time,
                nROIs=self.nROI,
                ROI_size=self.ROI_size,
                ROI_corners=corners,
            )
        return scene

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
            ).astype(int),
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


class VISDALevel2HDUList(VISDALevel1HDUList):
    filename = FORMATSDIR + "visda/level2_visda.xlsx"
    reference = VISDAReference


class VISDAFFILevel0HDUList(PandoraHDUList):
    filename = FORMATSDIR + "visda/level0-ffi_visda.xlsx"
    reference = VISDAReference

    def __to_l1__(self):
        return VISDAFFILevel1HDUList(self)


class VISDAFFILevel1HDUList(VISDAFFILevel0HDUList):
    filename = FORMATSDIR + "visda/level1-ffi_visda.xlsx"

    def get_scene(self):
        hdr = self[0].header
        prf = pa.SpatialPRF.from_reference()
        prf.imcorner = (hdr["ROISTRTX"], hdr["ROISTRTY"])
        prf.imshape = (hdr["ROISIZEY"], hdr["ROISIZEX"])
        scene = pa.SkyScene(prf, self.wcs, self.start_time)
        return scene

    def _append_scene_extensions(self):
        hdr = self[0].header
        scene = self.get_scene()
        self.append(scene.get_catalog_hdu())
        self.append(scene.get_prf_hdu())
        self.append(
            scene.get_aperture_hdu(
                SkyCoord(hdr["targ_ra"], hdr["targ_dec"], unit="deg")
            )
        )
        self[0].header["GAIA_ID"] = self["APERTURE"].header["GAIA_ID"]
        logger.info("Appended scene extensions")

    def __to_l2__(self):
        return VISDAFFILevel2HDUList(self)


class VISDAFFILevel2HDUList(VISDAFFILevel1HDUList):
    filename = FORMATSDIR + "visda/level2-ffi_visda.xlsx"
