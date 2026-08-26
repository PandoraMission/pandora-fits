"""Functions for processing data"""

# Standard library
import warnings
from copy import deepcopy

# Third-party
import astropy.units as u
import numpy as np
import pandoraaperture as pa

# from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from pandoraref import __version__ as prversion

from . import __version__, logger
from .database import AstrometryDataBase, Level0DataBase


class ProcessingMixins:
    def _get_reference_detector_image(self, name):
        ref = getattr(self.reference, f"get_{name}")()
        if isinstance(ref, u.Quantity):
            ref = ref.value
        if isinstance(ref, (int, float)):
            ref = np.ones((2048, 2048)) * ref
        if hasattr(self, "panel_row"):
            # value = np.asarray(
            #     [ref[r, :][:, c] for r, c in zip(self.star_row, self.star_column)]
            # )[None, :, :]
            # value = self.star_list_to_panels(value)
            value = ref[
                np.nan_to_num(self.panel_row).astype(int),
                np.nan_to_num(self.panel_column).astype(int),
            ] * (~self.border_mask).astype(ref.dtype)
        else:
            value = ref[self.row, :][:, self.column]
        return value

    def _apply_detector_image(self, name, func):
        if "COMMENT" not in self[0].header:
            self[0].header["COMMENT"] = ""
        if f"Subtracted {name}." in self[0].header["COMMENT"]:
            logger.warning(
                f"Attempted {name} application a second time. Ignoring {name} subtraction."
            )
            return
        value = self._get_reference_detector_image(name)
        self["SCIENCE"].data = func(self["SCIENCE"].data, value)
        self[0].header["COMMENT"] = f"Applied {name}."

    def _cast_to_float(self):
        logger.info("Casting data to float.")
        self[1] = fits.CompImageHDU(
            self[1].data.astype(float), header=self[1].header
        )
        self[0].header["COMMENT"] = "Cast data to float."

    def _subtract_stripes(self):
        logger.info("Subtracting stripes")
        if "FRMPCOAD" in self[0].header:
            self._apply_detector_image(
                "stripes", lambda x, y: x - (y * self.frmpcoad)
            )
        else:
            self._apply_detector_image("stripes", lambda x, y: x - y)

    def _subtract_bias(self):
        logger.info("Subtracting bias")
        if "FRMPCOAD" in self[0].header:
            self._apply_detector_image(
                "bias", lambda x, y: x - (y * self.frmpcoad)
            )
        else:
            self._apply_detector_image("bias", lambda x, y: x - y)

    def _subtract_dark(self):
        logger.info("Subtracting dark")
        self._apply_detector_image("dark", lambda x, y: x - y)

    def _divide_flat(self):
        logger.info("Dividing flat")

        def apply_flat(x, y):
            k = (x != 0).any(axis=0) & (y != 0).any(axis=0)
            x2 = deepcopy(x)
            if y.ndim == 2:
                x2[:, k] /= y[k]
            else:
                x2[:, k] /= y[:, k]
            return x2

        self._apply_detector_image("flat", apply_flat)

    def _multiply_gain(self):
        logger.info("Multiplying gain")
        self._apply_detector_image("gain", lambda x, y: x * y)
        self[1].header.set(
            "UNIT",
            "electrons/pixel  ",
            "data units: electrons/pixel",
        )

    def _append_quality(self):
        logger.info("Appending quality")
        quality = self._get_reference_detector_image("bad_pixel")
        self.append(fits.CompImageHDU(quality, name="QUALITY"))

    def _append_background(self):
        logger.info("Appending background")
        bkg = self.get_bkg()
        self.append(
            fits.CompImageHDU(
                bkg,
                name="BACKGROUND",
                header=fits.Header([self["SCIENCE"].header.cards["UNIT"]]),
            ),
        )

    def _append_wavelength(self):
        logger.info("Appending wavelength")
        if "VECTORS" in self:
            dy = self.row - np.nanmedian(self["VECTORS"].data["posy"])
        else:
            dy = self.row - np.ones(self[0].header["POSY"])
        wav = self.reference.get_wavelength_from_position(dy)
        sens = self.reference.get_spectrum_normalization_per_pixel(dy)
        tab = Table(
            data=[self.row * u.pixel, dy * u.pixel, wav, sens],
            names=["row", "drow", "wavelength", "sensitivity"],
        )
        hdu = fits.convenience.table_to_hdu(tab)
        hdu.header["EXTNAME"] = "WAVELENGTH_SOLUTION"
        self.append(hdu)

    def _append_wcs(self):
        logger.info("Applying WCS")
        if "ASTROMETRY" in self:
            logger.info("Applying WCS from astrometry extension")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, ra, dec, rot = self.astrometry
                k = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(rot)
                k &= (ra != 0) & (dec != 0) & (rot != 0)
                wcs = self.reference.get_wcs_from_VITL(
                    np.median(ra[k]), np.median(dec[k]), np.median(rot[k])
                )
        else:
            with warnings.catch_warnings():
                logger.info("Applying WCS from position information in header")
                warnings.simplefilter("ignore")
                hdr = self[0].header
                wcs = self.reference.get_wcs_from_VITL(
                    hdr["RA"] if "RA" in hdr else 0,
                    hdr["Dec"] if "Dec" in hdr else 0,
                    hdr["Roll"] if "Roll" in hdr else 0,
                )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wcs_hdr = wcs.to_header(relax=True)
            self[1].header.extend(wcs_hdr)
        x, y = wcs.world_to_pixel(self.coord)
        self[0].header["POSX"], self[0].header["POSY"] = (
            (float(x), "X [column] Pixel position of target"),
            (float(y), "Y [row] Pixel position of target"),
        )

        self[0].header["COMMENT"] = "Appended WCS."

    # def _append_wcs(self):
    #     df = self.get_position_data()
    #     k = df.visda_keepout.values
    #     wcs = self.reference.get_wcs_from_VITL(
    #         np.nanmedian(df.avg_ra.values[k]),
    #         np.nanmedian(df.avg_dec.values[k]),
    #         np.nanmedian(df.avg_rot.values[k]),
    #     )
    #     wcs_hdr = wcs.to_header(relax=True)
    #     self[1].header.extend(wcs_hdr)
    #     self[0].header["COMMENT"] = "Appended WCS."
    #     if "TEMP_TIME" in self:
    #         Table(self["TEMP_TIME"].data)

    def _append_vectors(self):
        df = self.get_position_data()

        # These need to be replaced with the instantaneous WCS
        # df["posx"], df["posy"] = self.wcs.world_to_pixel(
        #     SkyCoord(df.avg_ra.values, df.avg_dec.values, unit="deg")
        # )
        df["posx"] = np.nan
        df["posy"] = np.nan

        for tdx in range(len(df)):
            if (
                not np.isfinite(df.avg_ra[tdx])
                & np.isfinite(df.avg_dec[tdx])
                & np.isfinite(df.avg_rot[tdx])
            ):
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wcs = self.reference.get_wcs_from_VITL(
                    df.avg_ra[tdx], df.avg_dec[tdx], df.avg_rot[tdx]
                )
            df.loc[tdx, ["posx", "posy"]] = wcs.world_to_pixel(self.coord)

        posx, posy = self.wcs.world_to_pixel(self.coord)
        df["posx"] = df["posx"].fillna(posx)
        df["posy"] = df["posy"].fillna(posy)

        tab = fits.convenience.table_to_hdu(
            Table.from_pandas(df.fillna(np.nan))
        )
        tab.header.extend(fits.Header([("EXTNAME", "VECTORS", "")]))
        self.append(tab)
        self[0].header["COMMENT"] = "Appended Vectors."
        logger.info("Appended vectors.")

    def _append_error_extension(self):
        """Procedure to make error extension, specific to VISDALevel1HDUList."""
        # This is a silly estimate of photon noise, we're going to do better than this once we commission
        error = np.abs(self["science"].data) ** 0.5
        # This is one possible noise. Again, we'll do better after commissioning
        readnoise = (
            (
                (self.reference.get_readnoise() ** 2 * self.ncoadds) ** 0.5
                * u.pixel
            )
            .to(u.electron)
            .value
        )
        hdr = fits.Header([self["science"].header.cards["UNIT"]])
        errorhdu = fits.ImageHDU(error + readnoise, hdr, name="ERROR")
        self.append(errorhdu)

        logger.info("Appended error extension")
        return

    def _update_pointing_params(self):
        def update_attr(name, value, comment=None):
            if value is not None:
                if name in self[0].header:
                    if self[0].header[name] not in [None, ""]:
                        # Do not overwrite existing keywords
                        return
                self[0].header[name] = (value, comment)
            elif (value is None) & (name not in self[0].header):
                self[0].header[name] = (0, comment)

        time_range = (self.start_time.jd - 0.00001, self.end_time.jd + 0.00001)
        with Level0DataBase() as db:
            df = db.to_pandas(
                time_range=time_range,
                instrmnt=self.instrument,
                targ_id=self.targ_id,
            )
        if len(df) != 0:
            k = ~df["targ_ra"].isin([None])
            if k.any():
                targ_ra = df.loc[k, "targ_ra"].mode()[0]
            else:
                raise ValueError("No pointing information available")
                targ_ra = 0
            k = ~df["targ_dec"].isin([None])
            if k.any():
                targ_dec = df.loc[k, "targ_dec"].mode()[0]
            else:
                raise ValueError("No pointing information available")
                targ_dec = 0
            k = ~df["ra"].isin([None])
            if k.any():
                ra = df.loc[k, "ra"].mode()[0]
            else:
                raise ValueError("No pointing information available")

                ra = 0
            k = ~df["dec"].isin([None])
            if k.any():
                dec = df.loc[k, "dec"].mode()[0]
            else:
                raise ValueError("No pointing information available")

                dec = 0
            k = ~df["roll"].isin([None])
            if k.any():
                roll = df.loc[k, "roll"].mode()[0]
            else:
                raise ValueError("No pointing information available")
                roll = 0
        else:
            raise ValueError("No pointing information available")
            targ_ra, targ_dec, ra, dec, roll = 0, 0, 0, 0, 0

        if len(df) > 0:
            if len(df.targ_id.unique()) != 1:
                logger.warning(
                    "This file seems to cover multiple targets/pointings."
                )
            # This should select the most common target ID in the case of many target IDs
            targ_id = df.targ_id.mode()[0]
            # dpc_seq_id = df.dpc_seq_id.mode()[0]
            # start = df.start.mode()[0]
        else:
            targ_id = "unknown"
            # dpc_seq_id = "unknown"
            # start = 2454833
        update_attr(
            "TARG_RA",
            targ_ra,
            "Target right ascension [deg]",
        )
        update_attr(
            "TARG_DEC",
            targ_dec,
            "Target declination [deg]",
        )
        update_attr(
            "ra",
            ra,
            "Executed RA [deg]",
        )
        update_attr(
            "dec",
            dec,
            "Executed DEC [deg]",
        )
        update_attr(
            "roll",
            roll,
            "Executed roll [deg]",
        )
        update_attr(
            "TARG_ID",
            targ_id,
            "Target ID/keyword",
        )
        # update_attr(
        #     "DPCSEQID",
        #     dpc_seq_id,
        #     "DPC Obseravation Sequence ID",
        # )
        # update_attr(
        #     "SEQSTART",
        #     start,
        #     "DPC Observation Sequence Start",
        # )

    def _update_astrometry_extension(self):
        # For finding targets we tolerate any files that are taken during the same observation or within 30s of the observation.
        if "ASTROMETRY" in self:
            hdr = self["ASTROMETRY"].header
            self.pop("ASTROMETRY")
        else:
            hdr = None
        if "TEMP_TIME" in self:
            self.pop("TEMP_TIME")
        time_buffer = 30.0 / 86400.0
        time_range = (
            self.start_time.jd - time_buffer,
            self.end_time.jd + time_buffer,
        )
        with AstrometryDataBase() as db:
            df = db.to_pandas(
                time_range=time_range,
                columns=["jd", "ra", "dec", "roll", "temp"],
            )

        # Here we add the astrometry information to any file that can have it, but we won't overwrite it if it exists.
        ast_tab = df.rename(
            {
                "jd": "JD",
                "ra": "RightAscension",
                "dec": "Declination",
                "roll": "Rotation",
                "temp": "PCOTemp",
            },
            axis="columns",
        )
        ast_tab = fits.convenience.table_to_hdu(
            Table.from_pandas(ast_tab.fillna(np.nan))
        )
        ast_tab.header.extend(fits.Header([("EXTNAME", "ASTROMETRY", "")]))
        if hdr is not None:
            ast_tab.header.extend([h for h in hdr.cards])
        self.append(ast_tab)

    def _update_catalog_params(self):
        logger.info("Adding catalog header kwargs")
        cat = self.catalog_params
        self[0].header.extend(
            [
                (
                    "GAIADR3",
                    f"Gaia DR3 {cat['source_id']}",
                    "Gaia DR3 source ID",
                ),
                ("GAIARA", cat["ra"], "Gaia DR3 RA in J2016"),
                ("GAIADEC", cat["dec"], "Gaia DR3 DEC in J2016"),
                ("parallax", cat["parallax"], "Gaia DR3 parallax"),
                ("pmra", cat["pmra"], "Gaia DR3 pmra"),
                ("pmdec", cat["pmdec"], "Gaia DR3 pmdec"),
                (
                    "2MASSID",
                    f"2MASS J{cat['tmass_source_id']}",
                    "2MASS source ID",
                ),
                ("j_m", np.nan_to_num(cat["j_m"], -99), "2MASS j magnitude"),
                ("h_m", np.nan_to_num(cat["h_m"], -99), "2MASS h magnitude"),
                ("k_m", np.nan_to_num(cat["k_m"], -99), "2MASS k magnitude"),
                (
                    "g_m",
                    np.nan_to_num(cat["phot_g_mean_mag"], -99),
                    "Gaia DR3 g magnitude",
                ),
                (
                    "bp_m",
                    np.nan_to_num(cat["phot_bp_mean_mag"], -99),
                    "Gaia DR3 bp magnitude",
                ),
                (
                    "rp_m",
                    np.nan_to_num(cat["phot_rp_mean_mag"], -99),
                    "Gaia DR3 rp magnitude",
                ),
                (
                    "teff",
                    np.nan_to_num(cat["teff_gspphot"], -1),
                    "Gaia DR3 Teff",
                ),
                (
                    "logg",
                    np.nan_to_num(cat["logg_gspphot"], -1),
                    "Gaia DR3 logg",
                ),
            ]
        )

    def _append_aperture_extension(self):
        logger.info("Applying aperture extension")
        self.prf.imshape = (self[1].header["NAXIS2"], self[1].header["NAXIS1"])
        self.prf.imcorner = (
            self[0].header["ROISTRTY"],
            self[0].header["ROISTRTX"],
        )
        if self.instrument == "NIRDA":
            pixel_buffer = (230, 30)
        else:
            pixel_buffer = (30, 30)
        scene = pa.SkyScene(
            self.prf, self.wcs, self.start_time, pixel_buffer=pixel_buffer
        )

        # scene = pa.SkyScene(self.prf, self.wcs, self.start_time)

        if len(scene.cat) == 0:
            raise ValueError("Can not find catalog stars")
        # Delta pos is a bit of a hack, it seems like input PSF model is OBO
        hdu = scene.get_aperture_hdu(self.coord, delta_pos=(0, 0))
        self.append(hdu)
        hdu = scene.get_model_hdu(delta_pos=(0, 0))
        self.append(hdu)

    def to_level1(self, upcast=True, **kwargs):
        if self.level >= 1:
            raise ValueError("This is a Level 1 Product.")
        new = self.copy()
        new = new.fix()

        new[0].header["PFSOFTV"] = __version__
        new[0].header["PRSOFTV"] = prversion

        # This header keyword set isn't fitting in FITS conventions so we're renaming them if present.
        # We switch it out to "UNIT"
        for key in ["TTYPE1", "TFORM1", "TUNIT1"]:
            if key in new[1].header:
                new[1].header.remove(key)

        new[0].header["PFCLASS"] = new.__class__.__name__.replace(
            f"{self.level}", f"{self.level + 1}"
        )
        new[0].header["PFTIME"] = (
            Time.now().isot,
            "Pandora DPC Processing Time",
        )

        if upcast:
            new = new.__to_l1__()
        return new

    def to_level2(self, upcast=True, **kwargs):
        if self.level == 0:
            raise ValueError(
                "This is a Level 0 Product, convert to Level 1 first."
            )

        if self.level >= 2:
            raise ValueError("This is a Level 2 Product.")
        new = self.copy()
        new._update_pointing_params()
        new._append_wcs()
        new._update_astrometry_extension()
        new._update_catalog_params()

        if "VISDALevel1HDUList" in new.__class__.__name__:
            new = new.split_target()
            new.pop("STAR_TABLE")
            new[0].header["ROISTRTX"] = (
                new["ROI_TABLE"].data["StarROI_StartX"][0]
                + new[0].header["ROISTRTX"]
            )
            new[0].header["ROISTRTY"] = (
                new["ROI_TABLE"].data["StarROI_StartY"][0]
                + new[0].header["ROISTRTY"]
            )
            new[0].header["ROISIZEX"] = new[0].header["STARDIMS"]
            new[0].header["ROISIZEY"] = new[0].header["STARDIMS"]
            new.pop("ROI_TABLE")
            new._append_wcs()
            new._subtract_bias()
            new["SCIENCE"] = fits.CompImageHDU(
                new["SCIENCE"].data / new.exptime[:, None, None].value,
                name="SCIENCE",
                header=new["SCIENCE"].header,
            )
            new["SCIENCE"].header["UNIT"] = "ct / s"
            new._append_vectors()
            # vectors update wcs
            new._append_wcs()
            new._append_quality()

        elif "NIRDALevel1HDUList" in new.__class__.__name__:
            new = new.difference_sample()
            new._append_vectors()
            # vectors update wcs
            new._append_wcs()
            new._append_quality()
            new._append_wavelength()

        new._append_aperture_extension()
        new._append_background()
        new._score_file()
        new[0].header["PFSOFTV"] = __version__
        new[0].header["PRSOFTV"] = prversion
        new[0].header["PFCLASS"] = new.__class__.__name__.replace(
            f"{self.level}", f"{self.level + 1}"
        )
        new[0].header["PFTIME"] = (
            Time.now().isot,
            "Pandora DPC Processing Time",
        )

        if upcast:
            new = new.__to_l2__()
        return new

    # def to_level2(self, upcast=True, **kwargs):
    #     if self.level >= 2:
    #         raise ValueError("This is a Level 2 Product.")
    #     new = self.copy()
    #     new._cast_to_float()
    #     new._subtract_bias()
    #     new._subtract_dark()
    #     new._divide_flat()
    #     new._multiply_gain()
    #     new._append_quality()
    #     new._append_wcs()
    #     new._append_scene_extensions()
    #     new._append_error_extension()

    #     if self.instrument == "NIRDA":
    #         pix = new.row - new["catalog"].data["row"][0]
    #         wav = NIRDAReference.get_wavelength_position(pix)
    #         sens = NIRDAReference.get_spectrum_normalization_per_pixel(pix)

    #         wavtab = fits.TableHDU.from_columns(
    #             [
    #                 fits.Column(
    #                     "wavelength",
    #                     "D",
    #                     array=wav.value,
    #                     unit=wav.unit.to_string(),
    #                 ),
    #                 fits.Column(
    #                     "sensitivity",
    #                     "D",
    #                     array=sens.value,
    #                     unit=sens.unit.to_string(),
    #                 ),
    #             ],
    #             name="WAVELENGTH",
    #         )
    #         new.append(wavtab)

    #     new[0].header["PFCLASS"] = new.__class__.__name__.replace(
    #         f"{self.level}", f"{self.level + 1}"
    #     )
    #     new[0].header["PFTIME"] = (
    #         Time.now().isot,
    #         "Pandora DPC Processing Time",
    #     )

    #     if upcast:
    #         new = new.__to_l2__()
    #     return new
