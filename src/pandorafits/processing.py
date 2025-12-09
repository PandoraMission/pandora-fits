"""Functions for processing data"""

from datetime import timedelta

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.time import Time


class ProcessingMixins:
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
            hdr["ROISTRTX"] : hdr["ROISTRTX"] + hdr["ROISIZEX"],
            hdr["ROISTRTY"] : hdr["ROISTRTY"] + hdr["ROISIZEY"],
        ]
        return R, C

    @property
    def row(self):
        hdr = self[0].header
        return np.arange(hdr["ROISTRTX"], hdr["ROISTRTX"] + hdr["ROISIZEX"])

    @property
    def column(self):
        hdr = self[0].header
        return np.arange(hdr["ROISTRTY"], hdr["ROISTRTY"] + hdr["ROISIZEY"])

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

    def apply_quality(self):
        quality = self.reference.get_bad_pixel()[self.row][:, self.column]
        hdu = fits.CompImageHDU(quality, name="QUALITY")
        self.append(hdu)
        self[0].header["COMMENT"] = "Data quality extension created."
