# flake8: noqa W291
"""Database tools for MOC files database"""

import os
import sqlite3
import stat
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time

from .. import LEVEL0_DIR, __version__
from .mixins import DataBaseMixins


class Level0DataBase(DataBaseMixins):
    """Database for managing files that have been delivered by MOC."""

    def __init__(self):
        self.db_path = f"{LEVEL0_DIR}/pointings.db"
        self.conn = sqlite3.connect(self.db_path)
        self.cur = self.conn.cursor()

        self.cur.execute(
            """
        CREATE TABLE IF NOT EXISTS pointings (
            filename TEXT PRIMARY KEY,
            lvlfilepath TEXT,
            dir TEXT,
            crsoftver TEXT,
            pfsoftver TEXT,
            finetime INT,
            corstime INT,
            jd FLOAT,
            date STR,
            exptime FLOAT,
            dpc_obs_id INT,
            start FLOAT,
            instrmnt TEXT,
            roisizex INT,
            roisizey INT,
            roistrtx INT,
            roistrty INT,
            next INT,
            astrometry BOOL,
            targ_id STR,
            ra FLOAT,
            dec FLOAT,
            naxis1 INT,
            naxis2 INT,
            naxis3 INT,
            naxis4 INT,
            badchecksum INT,
            baddatasum INT,
            filesize FLOAT
        )
        """
        )

        self.update_str = """INSERT INTO pointings 
        (filename, lvlfilepath, dir, crsoftver, pfsoftver, finetime, corstime,
        jd, date, exptime, dpc_obs_id, start, instrmnt, roisizex,
        roisizey, roistrtx, roistrty, next, astrometry,
        targ_id, ra, dec, naxis1, naxis2, naxis3, naxis4, 
        badchecksum, baddatasum, filesize)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        self.conn.commit()

        os.chmod(
            self.db_path,
            stat.S_IRUSR
            | stat.S_IWUSR  # owner: read/write
            | stat.S_IRGRP
            | stat.S_IWGRP,  # group: read/write
        )

    def __repr__(self):
        return "Pandora Level0DataBase"

    def check_filename_in_database(self, filename):
        self.cur.execute(
            "SELECT 1 FROM pointings WHERE filename=?",
            ((filename.split("/")[-1] if "/" in filename else filename),),
        )
        return self.cur.fetchone() is not None

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
                for key in ["FINETIME", "CORSTIME", "INSTRMNT"]:
                    if key not in hdr:
                        return
                return (
                    filename.split("/")[-1],
                    filename,
                    "/".join(filename.split("/")[:-1]),
                    hdr["CRSOFTV"],
                    __version__,
                    hdr["FINETIME"],
                    hdr["CORSTIME"],
                    time.jd,
                    time.isot,
                    exptime,
                    -1,
                    -1,
                    hdr["INSTRMNT"],
                    hdr["ROISIZEX"],
                    hdr["ROISIZEY"],
                    hdr["ROISTRTX"],
                    hdr["ROISTRTY"],
                    len(hdulist),
                    "ASTROMETRY"
                    in np.asarray([hdu.header["extname"] for hdu in hdulist]),
                    hdr["TARG_ID"] if "TARG_ID" in hdr else None,
                    hdr["TARG_RA"] if "TARG_RA" in hdr else None,
                    hdr["TARG_DEC"] if "TARG_DEC" in hdr else None,
                    hdr1["NAXIS1"] if "NAXIS1" in hdr1 else None,
                    hdr1["NAXIS2"] if "NAXIS2" in hdr1 else None,
                    hdr1["NAXIS3"] if "NAXIS3" in hdr1 else None,
                    hdr1["NAXIS4"] if "NAXIS4" in hdr1 else None,
                    badchecksum,
                    baddatasum,
                    filesize,
                )

    def crawl_and_add(self, root):
        for image_type in ["InfImg", "VisSci", "VisImg"]:
            # for path in Path(root).rglob(f"*{image_type}*.fits"):
            #     self.add_entry(self.get_entry(str(path)))
            paths = [
                str(path)
                for path in Path(root).rglob(f"*{image_type}*.fits")
                if not self.check_filename_in_database(str(path))
            ]
            rows = []
            for path in paths:
                values = self.get_entry(path)
                if values is not None:
                    rows.append(values)
            self.add_entries(rows)
        self.update_pointings()

    # def crawl_and_add_parallel(self, root, max_workers=16):
    #     for image_type in ["InfImg", "VisSci", "VisImg"]:
    #         # for path in Path(root).rglob(f"*{image_type}*.fits"):
    #         #     self.add_entry(self.get_entry(str(path)))
    #         paths = [
    #             str(path)
    #             for path in Path(root).rglob(f"*{image_type}*.fits")
    #             if not self.check_filename_in_database(str(path))
    #         ]
    #         rows = []
    #         with ThreadPoolExecutor(max_workers=max_workers) as ex:
    #             futures = [ex.submit(self.get_entry, p) for p in paths]
    #             for fut in as_completed(futures):
    #                 row = fut.result()
    #                 if row is not None:
    #                     rows.append(row)
    #         self.add_entries(rows)
    #     self.update_pointings()

    def _update_dpc_obs_id(self):
        sql = """
        WITH changes AS (
        SELECT
            targ_id,
            CASE
            WHEN targ_id = LAG(targ_id) OVER (ORDER BY jd, targ_id)
            THEN 0          -- same target as previous row → same visit
            ELSE 1          -- target changed (or first row) → new visit
            END AS is_new_visit
        FROM pointings
        ),
        visits AS (
        SELECT
            targ_id,
            SUM(is_new_visit) OVER (ORDER BY jd, targ_id) AS dpc_obs_id
        FROM changes
        )
        UPDATE pointings
        SET dpc_obs_id = (
        SELECT dpc_obs_id FROM visits WHERE visits.targ_id = pointings.targ_id
        );

        """
        self.conn.execute(sql)

    def _update_target(self):
        sql = """
        WITH filled AS (
        SELECT
            targ_id,
            MAX(ra)  OVER (
            PARTITION BY dpc_obs_id
            ORDER BY jd
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS ra_filled,
            MAX(dec) OVER (
            PARTITION BY dpc_obs_id
            ORDER BY jd
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS dec_filled
        FROM pointings
        )
        UPDATE pointings
        SET ra  = (SELECT ra_filled  FROM filled WHERE filled.targ_id = pointings.targ_id),
            dec = (SELECT dec_filled FROM filled WHERE filled.targ_id = pointings.targ_id)
        WHERE ra IS NULL OR dec IS NULL;
        """
        self.conn.execute(sql)

    def _update_start(self):
        sql = """
        WITH filled AS (
        SELECT
            targ_id,
            MIN(jd) OVER (
            PARTITION BY dpc_obs_id
            ORDER BY jd
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS start_filled
        FROM pointings
        )
        UPDATE pointings
        SET start = (SELECT start_filled FROM filled WHERE filled.targ_id = pointings.targ_id)
        """
        self.conn.execute(sql)

    def update_pointings(self):
        self._update_dpc_obs_id()
        self._update_target()
        self._update_start()
