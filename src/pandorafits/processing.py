"""Functions for processing data"""

# Standard library
from copy import deepcopy
from functools import cached_property

# Third-party
import astropy.units as u
import gaiaoffline
import numpy as np
from astropy.coordinates import Distance, SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

from . import NIRDAReference, __version__, logger, ps
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
        self._apply_detector_image(
            "stripes", lambda x, y: x - (y * self.frmpcoad)
        )

    def _subtract_bias(self):
        logger.info("Subtracting bias")
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

    # def _append_wcs(self):
    #     logger.info("Applying WCS")
    #     hdr = self[0].header
    #     wcs = self.reference.get_wcs(
    #         hdr["targ_ra"] if hdr["targ_ra"] is not None else 0,
    #         hdr["targ_dec"] if hdr["targ_dec"] is not None else 0,
    #         hdr["targ_rll"] if hdr["targ_rll"] is not None else 40,
    #         distortion=True,
    #         yreflect=True,
    #     )
    #     wcs_hdr = wcs.to_header(relax=True)

    #     self[1].header.extend(wcs_hdr)
    #     self[0].header["COMMENT"] = "Appended WCS."

    def _append_wcs(self):
        k = self.get_earth_angle() > self.visda_keepout
        df = self.get_position_data()
        wcs = self.reference.get_wcs_from_VITL(
            np.median(df.avg_ra.values[k]),
            np.median(df.avg_dec.values[k]),
            np.median(df.avg_rot.values[k]),
        )
        wcs_hdr = wcs.to_header(relax=True)
        self[1].header.extend(wcs_hdr)
        self[0].header["COMMENT"] = "Appended WCS."
        if "TEMP_TIME" in self:
            Table(self["TEMP_TIME"].data)

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

        time_range = (self.start_time.jd, self.end_time.jd)
        with Level0DataBase() as db:
            df = db.to_pandas(time_range=time_range)

        if len(df) != 0:
            k = ~df["targ_ra"].isin([None])
            if k.any():
                targ_ra = df.loc[k, "targ_ra"].mode()[0]
            else:
                targ_ra = 0
            k = ~df["targ_dec"].isin([None])
            if k.any():
                targ_dec = df.loc[k, "targ_dec"].mode()[0]
            else:
                targ_dec = 0
            k = ~df["targ_rll"].isin([None])
            if k.any():
                targ_rll = df.loc[k, "targ_rll"].mode()[0]
            else:
                targ_rll = 40
        else:
            targ_ra, targ_dec, targ_rll = 0, 0, 40

        if len(df) > 0:
            if len(df.targ_id.unique()) != 1:
                logger.warning(
                    "This file seems to cover multiple targets/pointings."
                )
            # This should select the most common target ID in the case of many target IDs
            targ_id = df.targ_id.mode()[0]
            dpc_seq_id = df.dpc_seq_id.mode()[0]
            start = df.start.mode()[0]
        else:
            targ_id = "unknown"
            dpc_seq_id = "unknown"
            start = 2454833

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
            "TARG_RLL",
            targ_rll,
            "Commanded roll [deg]",
        )
        update_attr(
            "TARG_ID",
            targ_id,
            "Target ID/keyword",
        )
        update_attr(
            "DPCSEQID",
            dpc_seq_id,
            "DPC Obseravation Sequence ID",
        )
        update_attr(
            "SEQSTART",
            start,
            "DPC Observation Sequence Start",
        )

    def _update_astrometry_extension(self):
        # For finding targets we tolerate any files that are taken during the same observation or within 30s of the observation.
        if "ASTROMETRY" not in self:
            time_buffer = 30.0 / 86400.0
            time_range = (
                self.start_time.jd - time_buffer,
                self.end_time.jd + time_buffer,
            )
            with AstrometryDataBase() as db:
                df = db.to_pandas(time_range=time_range)

            # Here we add the astrometry information to any file that can have it, but we won't overwrite it if it exists.
            ast_tab = df[["jd", "ra", "dec", "roll"]].rename(
                {
                    "jd": "JD",
                    "ra": "RightAscension",
                    "dec": "Declination",
                    "roll": "Rotation",
                },
                axis="columns",
            )
            ast_tab = fits.convenience.table_to_hdu(
                Table.from_pandas(ast_tab.fillna(np.nan))
            )
            ast_tab.header.extend(fits.Header([("EXTNAME", "ASTROMETRY", "")]))
            self.append(ast_tab)
        else:
            return

    @cached_property
    def catalog_params(self):
        with gaiaoffline.Gaia(
            tmass_crossmatch=True, photometry_output="magnitude"
        ) as gaia:
            df = gaia.conesearch(self.targ_ra, self.targ_dec, 0.1).reset_index(
                drop=True
            )

        coords = SkyCoord(
            ra=df["ra"].values * u.deg,
            dec=df["dec"].values * u.deg,
            pm_ra_cosdec=df["pmra"].fillna(0).values * u.mas / u.year,
            pm_dec=df["pmdec"].fillna(0).values * u.mas / u.year,
            obstime=Time.strptime("2016", "%Y"),
            distance=Distance(
                parallax=df["parallax"].fillna(0).values * u.mas,
                allow_negative=True,
            ),
            radial_velocity=df["radial_velocity"].fillna(0).values
            * u.km
            / u.s,
        ).apply_space_motion(self.start_time)
        df = df.iloc[coords.separation(self.coord).argmin()]
        return df.to_dict()

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
                ("j_m", cat["j_m"], "2MASS j magnitude"),
                ("h_m", cat["h_m"], "2MASS h magnitude"),
                ("k_m", cat["k_m"], "2MASS k magnitude"),
                ("g_m", cat["phot_g_mean_mag"], "Gaia DR3 g magnitude"),
                ("bp_m", cat["phot_bp_mean_mag"], "Gaia DR3 bp magnitude"),
                ("rp_m", cat["phot_rp_mean_mag"], "Gaia DR3 rp magnitude"),
            ]
        )

    def to_level1(self, upcast=True, **kwargs):
        if self.level >= 1:
            raise ValueError("This is a Level 1 Product.")

        new = self.copy()

        new[0].header["PFSOFTV"] = __version__

        # This header keyword set isn't fitting in FITS conventions so we're renaming them if present.
        # We switch it out to "UNIT"
        for key in ["TTYPE1", "TFORM1", "TUNIT1"]:
            if key in new[1].header:
                new[1].header.remove(key)

        new[1].header["UNIT"] = "COUNTS"
        new[1].header["PFSOFTV"] = __version__
        new._update_pointing_params()
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
        if self.level >= 2:
            raise ValueError("This is a Level 2 Product.")
        new = self.copy()
        new._cast_to_float()
        new._subtract_bias()
        new._subtract_dark()
        new._divide_flat()
        new._multiply_gain()
        new._append_quality()
        new._append_wcs()
        new._append_scene_extensions()
        new._append_error_extension()

        if self.instrument == "NIRDA":
            pix = new.row - new["catalog"].data["row"][0]
            wav = NIRDAReference.get_wavelength_position(pix)
            sens = NIRDAReference.get_spectrum_normalization_per_pixel(pix)

            wavtab = fits.TableHDU.from_columns(
                [
                    fits.Column(
                        "wavelength",
                        "D",
                        array=wav.value,
                        unit=wav.unit.to_string(),
                    ),
                    fits.Column(
                        "sensitivity",
                        "D",
                        array=sens.value,
                        unit=sens.unit.to_string(),
                    ),
                ],
                name="WAVELENGTH",
            )
            new.append(wavtab)

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

    @cached_property
    def earth_illumination(self):
        return ps.get_earth_illumination(self.time)

    @cached_property
    def visda_keepout(self):
        earth_illum = self.earth_illumination
        keepout = np.ones(len(earth_illum)) * 75
        keepout[earth_illum.value < 90] = (
            2.5 * (90 - earth_illum[earth_illum.value < 90].value) + 105
        )
        keepout[keepout > 135] = 135
        return keepout * u.deg

    @cached_property
    def nirda_keepout(self):
        earth_illum = self.earth_illumination
        keepout = np.ones(len(earth_illum)) * 75
        keepout[earth_illum.value < 90] = (
            2.5 * (90 - earth_illum[earth_illum.value < 90].value) + 80.5
        )
        keepout[keepout > 135] = 135
        return keepout * u.deg
