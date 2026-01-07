"""Functions for processing data"""

from copy import deepcopy
from datetime import timedelta

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy.table import Table

from . import __version__, logger
from .database import AstrometryDataBase


class ProcessingMixins:
    @property
    def wcs(self):
        return [
            WCS(self[idx].header[3:])
            for idx in range(len(self))
            if "WCSAXES" in self[idx].header
        ][0]

    @property
    def start_time(self):
        """Given Pandora HDUList obtains the detector time in TAI."""
        time = (
            Time("2000-01-01 12:00:00", scale="tai")
            + timedelta(
                seconds=self[0].header["CORSTIME"],
                milliseconds=self[0].header["FINETIME"] / 1e6,
            )
        ).utc
        return time

    @property
    def pixel_coordinates(self):
        hdr = self[0].header
        R, C = np.mgrid[
            hdr["ROISTRTY"] : hdr["ROISTRTY"] + hdr["ROISIZEY"],
            hdr["ROISTRTX"] : hdr["ROISTRTX"] + hdr["ROISIZEX"],
        ]
        return R, C

    @property
    def row(self):
        hdr = self[0].header
        return np.arange(hdr["ROISTRTY"], hdr["ROISTRTY"] + hdr["ROISIZEY"])

    @property
    def column(self):
        hdr = self[0].header
        return np.arange(hdr["ROISTRTX"], hdr["ROISTRTX"] + hdr["ROISIZEX"])

    @property
    def frame_time(self):
        if "FRMTIME" in self[0].header:
            return (u.millisecond * self[0].header["FRMTIME"]).to(u.second)
        elif "EXPTIMEU" in self[0].header:
            return (
                u.microsecond
                * self[0].header["EXPTIMEU"]
                * self[0].header["FRMSCLCT"]
                / self[1].header["NAXIS3"]
            ).to(u.second)
        elif "EXPTIME" in self[0].header:
            return (
                u.microsecond
                * self[0].header["EXPTIME"]
                * self[0].header["FRMSCLCT"]
                / self[1].header["NAXIS3"]
            ).to(u.second)

    @property
    def nframes(self):
        return self[1].header[f"NAXIS{self[1].header['NAXIS']}"]

    @property
    def ncoadds(self):
        if "FRMPCOAD" in self[0].header:
            return self[0].header["FRMPCOAD"]
        elif self[0].header["INSTRMNT"] == "VISDA":
            return 1
        if self[0].header["GRPSAVGD"] == 0:
            return 1
        else:
            logger.warning("Currently missing GROUPS keyword header, check with Lance")
            return 1

    @property
    def end_time(self):
        return self.start_time + self.nframes * self.frame_time

    def time(self):
        dt = timedelta(seconds=self.frame_time.to(u.second).value)
        return (self.start_time + (np.arange(self.nframes) * dt)).jd

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
        if f"Subtracted {name}." in self[0].header["COMMENT"]:
            logger.warning(
                f"Attempted {name} application a second time. Ignoring {name} subtraction."
            )
            return
        value = self._get_reference_detector_image(name)
        self["SCIENCE"].data = func(self["SCIENCE"].data, value)
        self[0].header["COMMENT"] = f"Applied {name}."

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

    def _append_wcs(self):
        logger.info("Applying WCS")
        hdr = self[0].header
        wcs = self.reference.get_wcs(
            hdr["targ_ra"] if hdr["targ_ra"] is not None else 0,
            hdr["targ_dec"] if hdr["targ_dec"] is not None else 0,
            hdr["targ_rll"] if hdr["targ_rll"] is not None else 40,
            distortion=True,
            yreflect=True,
        )
        wcs_hdr = wcs.to_header(relax=True)

        self[1].header.extend(wcs_hdr)
        self[0].header["COMMENT"] = "Appended WCS."

    def _append_error_extension(self):
        """Procedure to make error extension, specific to VISDALevel1HDUList."""
        # This is a silly estimate of photon noise, we're going to do better than this once we commission
        error = np.abs(self["science"].data) ** 0.5
        # This is one possible noise. Again, we'll do better after commissioning
        readnoise = (
            ((self.reference.get_readnoise() ** 2 * self.ncoadds) ** 0.5 * u.pixel)
            .to(u.electron)
            .value
        )
        hdr = fits.Header([self["science"].header.cards["UNIT"]])
        errorhdu = fits.ImageHDU(error + readnoise, hdr, name="ERROR")
        self.append(errorhdu)

        logger.info("Appended error extension")
        return

    def to_level1(self, upcast=True, **kwargs):
        new = self.copy()
        new[0].header["PFSOFTV"] = __version__

        def update_attr(name, value, comment=None):
            if value is not None:
                if name in new[0].header:
                    if new[0].header[name] not in [None, ""]:
                        # Do not overwrite existing keywords
                        return
                new[0].header[name] = (value, comment)
            elif (value is None) & (name not in new[0].header):
                new[0].header[name] = (0, comment)

        # For finding targets we tolerate any files that are taken during the same observation or within 30s of the observation.
        time_buffer = 30.0 / 86400.0
        time_range = (self.start_time.jd - time_buffer, self.end_time.jd + time_buffer)
        with AstrometryDataBase() as db:
            df = db.to_pandas(time_range=time_range)

        if len(df) != 0:
            targ_ra, targ_dec, targ_rll = (
                df.targ_ra.mode()[0],
                df.targ_dec.mode()[0],
                df.targ_rll.mode()[0],
            )
        else:
            targ_ra, targ_dec, targ_rll = 0, 0, 40

        if len(df) > 0:
            if len(df.targ_id.unique()) != 1:
                logger.warning("This file seems to cover multiple targets/pointings.")
            # This should select the most common target ID in the case of many target IDs
            targ_id = df.targ_id.mode()[0]
        else:
            targ_id = "unknown"

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

        # This header keyword set isn't fitting in FITS conventions so we're renaming them if present.
        # We switch it out to "UNIT"
        for key in ["TTYPE1", "TFORM1", "TUNIT1"]:
            if key in new[1].header:
                new[1].header.remove(key)

        new[1].header["UNIT"] = "COUNTS"
        new[1].header["PFSOFTV"] = __version__

        if "ASTROMETRY" not in self:
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
            ast_tab = fits.convenience.table_to_hdu(Table.from_pandas(ast_tab))
            ast_tab.header.extend(fits.Header([("EXTNAME", "ASTROMETRY", "")]))
            new.append(ast_tab)

        new[0].header["PFCLASS"] = new.__class__.__name__.replace(
            f"{self.level}", f"{self.level + 1}"
        )
        if upcast:
            new = new.__to_l1__()
        return new

    def to_level2(self, upcast=True, **kwargs):
        if "2" in self.__class__.__name__:
            raise ValueError("This is a Level 2 Product.")
        new = self.copy()
        new[1] = fits.CompImageHDU(new[1].data.astype(float), header=new[1].header)
        new[0].header["COMMENT"] = "Cast data to float."
        new._subtract_bias()
        new._subtract_dark()
        new._divide_flat()
        new._multiply_gain()
        new._append_quality()
        new._append_wcs()
        new._append_scene_extensions()
        new._append_error_extension()
        new[0].header["PFCLASS"] = new.__class__.__name__.replace(
            f"{self.level}", f"{self.level + 1}"
        )

        if upcast:
            new = new.__to_l2__()
        return new
