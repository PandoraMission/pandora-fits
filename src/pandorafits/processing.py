"""Functions for processing data"""

from copy import deepcopy
from datetime import timedelta

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from . import __version__, logger


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
        self[1].data = func(self[1].data, value)
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

    def _append_quality(self):
        logger.info("Appending quality")
        quality = self._get_reference_detector_image("bad_pixel")
        self.append(fits.CompImageHDU(quality, name="QUALITY"))

    def _append_wcs(self):
        logger.info("Applying WCS")
        hdr = self[0].header
        wcs = self.reference.get_wcs(
            hdr["targ_ra"],
            hdr["targ_dec"],
            hdr["targ_rll"],
            distortion=True,
            yreflect=True,
        )
        self[1].header.extend(wcs.to_header(relax=True))
        self[0].header["COMMENT"] = "Appended WCS."

    def to_level1(
        self, targ_ra=None, targ_dec=None, targ_rll=None, upcast=True, **kwargs
    ):
        if "1" in self.__class__.__name__:
            raise ValueError("This is a Level 1 Product.")
        new = self.copy()
        new[0].header["PFSOFTV"] = __version__

        def update_attr(name, value, comment=None):
            if value is not None:
                new[0].header[name] = (value, comment)
            elif (value is not None) & (name not in new[0].header):
                new[0].header[name] = (0, comment)

        update_attr("TARG_RA", targ_ra, "Target right ascension [deg]")
        update_attr("TARG_DEC", targ_dec, "Target declination [deg]")
        update_attr("TARG_RLL", targ_rll, "Commanded roll [deg]")
        if upcast:
            return new.__to_l1__()
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
        if upcast:
            return new.__to_l2__()
        return new
