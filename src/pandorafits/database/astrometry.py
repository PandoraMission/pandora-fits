# flake8: noqa W291
"""Database tools for holding astrometry as measured on board."""

import os
import warnings
from datetime import timedelta
from pathlib import Path

from astropy.io import fits
from astropy.time import Time

from .. import DATA_DIR, __version__
from .level0 import Level0DataBase
from .mixins import DataBaseMixins


class AstrometryDataBase(Level0DataBase, DataBaseMixins):
    """Database for managing astrometry of Pandora"""

    table_name = "astrometry"
    _sql_key_dict = {
        "filename": "TEXT",
        "dir": "TEXT",
        "crsoftver": "TEXT",
        "pfsoftver": "TEXT",
        "jd": "FLOAT",
        "exptime": "FLOAT",
        "dpc_obs_id": "INT",
        "start": "FLOAT",
        "targ_id": "STR",
        "targ_ra": "FLOAT",
        "targ_dec": "FLOAT",
        "targ_rll": "FLOAT",
        "ra": "FLOAT",
        "dec": "FLOAT",
        "roll": "FLOAT",
        "temp": "FLOAT",
        "badchecksum": "INT",
        "baddatasum": "INT",
        "filesize": "FLOAT",
    }

    def __repr__(self):
        return "Pandora AstrometryDataBase"

    def get_entry(self, filename):
        filesize = os.path.getsize(filename) / (1024 * 1024)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # capture all warnings

            with fits.open(filename, lazy_load_hdus=True) as hdulist:
                if len(hdulist) <= 1:
                    return
                badchecksum = len(
                    [warn for warn in w if "Checksum" in str(warn.message)]
                )
                baddatasum = len([warn for warn in w if "Datasum" in str(warn.message)])

                hdr = hdulist[0].header
                time = (
                    Time("2000-01-01 12:00:00", scale="tai")
                    + timedelta(
                        seconds=hdr["CORSTIME"],
                        milliseconds=hdr["FINETIME"] / 1e6,
                    )
                ).utc
                hdr1 = hdulist[1].header

                if "FRMTIME" in hdr:
                    frame_time = hdr["FRMTIME"] / 1000
                elif "EXPTIMEU" in hdr:
                    frame_time = (
                        hdr["EXPTIMEU"] * hdr["FRMSCLCT"] / hdr1["NAXIS3"]
                    ) / 1.0e6
                elif "EXPTIME" in hdr:
                    frame_time = (
                        hdr["EXPTIME"] * hdr["FRMSCLCT"] / hdr1["NAXIS3"]
                    ) / 1.0e6

                nframes = hdr1[f"NAXIS{hdr1['NAXIS']}"]
                exptime = nframes * frame_time
                for key in ["FINETIME", "CORSTIME"]:
                    if key not in hdr:
                        return
                for ext in ["ASTROMETRY", "TEMP_TIME"]:
                    if ext not in hdulist:
                        return

                astrometry_data = hdulist["ASTROMETRY"].data
                temp_data = hdulist["TEMP_TIME"].data
                return [
                    (
                        filename.split("/")[-1],
                        "/".join(filename.split("/")[:-1]),
                        hdr["CRSOFTV"],
                        __version__,
                        time.jd + ((temp_data[idx][0] / 1e3) / (24.0 * 60.0 * 60.0)),
                        exptime,
                        -1,
                        -1,
                        hdr["TARG_ID"] if "TARG_ID" in hdr else None,
                        hdr["TARG_RA"] if "TARG_RA" in hdr else None,
                        hdr["TARG_DEC"] if "TARG_DEC" in hdr else None,
                        0.0,
                        astrometry_data[idx][0],
                        astrometry_data[idx][1],
                        astrometry_data[idx][2],
                        temp_data[idx][1],
                        badchecksum,
                        baddatasum,
                        filesize,
                    )
                    for idx in range(len(astrometry_data))
                ]

    def crawl_and_add(self):
        root = DATA_DIR
        for image_type in ["VisSci"]:
            paths = [
                str(path)
                for path in Path(root).rglob(f"*{image_type}*.fits")
                if not self.check_filename_in_database(str(path))
            ]
            rows = []
            for path in paths:
                values = self.get_entry(path)
                if values is not None:
                    [rows.append(v) for v in values]
            self.add_entries(rows)
        self.update_pointings()

    def update_pointings(self):
        self._update_dpc_obs_id()
        self._update_start()
        self._update_roll()
