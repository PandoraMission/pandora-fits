# flake8: noqa W291
"""Database tools for MOC files database"""

import os
import sqlite3
import stat
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time

from .. import DATA_DIR, LEVEL0_DIR, __version__
from ..roll import get_roll
from .mixins import DataBaseMixins
from .astrometry import AstrometryDataBase
from .targets import TargetDataBase


class Level0DataBase(DataBaseMixins):
    """Database for managing files that have been delivered by MOC."""

    table_name = "pointings"
    db_path = f"{LEVEL0_DIR}/level0.db"
    level = 0
    _sql_key_dict = {
        "filename": "TEXT PRIMARY KEY",
        "lvlfilename": "TEXT",
        "dir": "TEXT",
        "lvldir": "TEXT",
        "crsoftver": "TEXT",
        "pfsoftver": "TEXT",
        "finetime": "INT",
        "corstime": "INT",
        "jd": "FLOAT",
        "date": "STR",
        "exptime": "FLOAT",
        "dpc_obs_id": "INT",
        "dpc_hash_key": "TEXT",
        "start": "FLOAT",
        "instrmnt": "TEXT",
        "roisizex": "INT",
        "roisizey": "INT",
        "roistrtx": "INT",
        "roistrty": "INT",
        "next": "INT",
        "astrometry": "BOOL",
        "targ_id": "STR",
        "targ_ra": "FLOAT",
        "targ_dec": "FLOAT",
        # "targ_rll": "FLOAT",
        "naxis1": "INT",
        "naxis2": "INT",
        "naxis3": "INT",
        "naxis4": "INT",
        "badchecksum": "INT",
        "baddatasum": "INT",
        "filesize": "FLOAT",
    }

    # def __init__(self):
    #     self.conn = sqlite3.connect(self.db_path)
    #     self.cur = self.conn.cursor()

    #     key_string = ", ".join(
    #         [f"{key} {item}" for key, item in self._sql_key_dict.items()]
    #     )
    #     self.cur.execute(
    #         f"""
    #     CREATE TABLE IF NOT EXISTS {self.table_name} ({key_string})
    #     """
    #     )

    #     key_string = ", ".join([f"{key}" for key, item in self._sql_key_dict.items()])
    #     value_string = ", ".join(["?"] * len(self._sql_key_dict))
    #     self.update_str = f"""INSERT INTO {self.table_name}
    #     ({key_string})
    #     VALUES ({value_string})"""
    #     self.conn.commit()

    #     os.chmod(
    #         self.db_path,
    #         stat.S_IRUSR
    #         | stat.S_IWUSR  # owner: read/write
    #         | stat.S_IRGRP
    #         | stat.S_IWGRP,  # group: read/write
    #     )

    def __repr__(self):
        return f"Pandora Level{self.level}DataBase"

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
                    filename.split("/")[-1],
                    "/".join(filename.split("/")[:-1]),
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
                    # 40.0,
                    hdr1["NAXIS1"] if "NAXIS1" in hdr1 else None,
                    hdr1["NAXIS2"] if "NAXIS2" in hdr1 else None,
                    hdr1["NAXIS3"] if "NAXIS3" in hdr1 else None,
                    hdr1["NAXIS4"] if "NAXIS4" in hdr1 else None,
                    badchecksum,
                    baddatasum,
                    filesize,
                )

    def crawl_and_add(self):
        root = DATA_DIR
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

    def _update_dpc_obs_id(self):
        sql = f"""
        WITH changes AS (
        SELECT
            targ_id,
            CASE
            WHEN targ_id = LAG(targ_id) OVER (ORDER BY jd, targ_id)
            THEN 0          -- same target as previous row → same visit
            ELSE 1          -- target changed (or first row) → new visit
            END AS is_new_visit
        FROM {self.table_name}
        ),
        visits AS (
        SELECT
            targ_id,
            SUM(is_new_visit) OVER (ORDER BY jd, targ_id) AS dpc_obs_id
        FROM changes
        )
        UPDATE {self.table_name}
        SET dpc_obs_id = (
        SELECT dpc_obs_id FROM visits WHERE visits.targ_id = {self.table_name}.targ_id
        );

        """
        self.conn.execute(sql)

    def _update_target(self):
        sql = f"""
        WITH filled AS (
        SELECT
            targ_id,
            MAX(targ_ra)  OVER (
            PARTITION BY dpc_obs_id
            ORDER BY jd
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS ra_filled,
            MAX(targ_dec) OVER (
            PARTITION BY dpc_obs_id
            ORDER BY jd
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS dec_filled
        FROM {self.table_name}
        )
        UPDATE {self.table_name}
        SET targ_ra  = (SELECT ra_filled  FROM filled WHERE filled.targ_id = {self.table_name}.targ_id),
            targ_dec = (SELECT dec_filled FROM filled WHERE filled.targ_id = {self.table_name}.targ_id)
        WHERE targ_ra IS NULL OR targ_dec IS NULL;
        """
        self.conn.execute(sql)

    def _update_start(self):
        sql = f"""
        WITH filled AS (
        SELECT
            targ_id,
            MIN(jd) OVER (
            PARTITION BY dpc_obs_id
            ORDER BY jd
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS start_filled
        FROM {self.table_name}
        )
        UPDATE {self.table_name}
        SET start = (SELECT start_filled FROM filled WHERE filled.targ_id = {self.table_name}.targ_id)
        """
        self.conn.execute(sql)

    # def _update_roll(self):
    #     df = pd.read_sql_query(
    #         f"SELECT start, targ_ra, targ_dec FROM {self.table_name} WHERE start = jd",
    #         self.conn,
    #     )
    #     df["targ_rll"] = [
    #         get_roll(Time(start, format="jd"), SkyCoord(ra, dec, unit="deg"))[0].value
    #         for start, ra, dec in df.values
    #     ]

    #     # 1) write df to a temp table
    #     df.to_sql("roll_map", self.conn, if_exists="replace", index=False)

    #     # 2) (optional but strongly recommended) index the join keys in both tables
    #     self.cur.execute(
    #         f"CREATE INDEX IF NOT EXISTS idx_main_keys ON {self.table_name}(start, targ_ra, targ_dec)"
    #     )
    #     self.cur.execute(
    #         "CREATE INDEX IF NOT EXISTS idx_map_keys  ON roll_map(start, targ_ra, targ_dec)"
    #     )
    #     self.conn.commit()

    #     # 3) update matching rows (fills duplicates in main table too)
    #     self.cur.execute(
    #         f"""
    #     UPDATE {self.table_name}
    #     SET targ_rll = (
    #     SELECT m.targ_rll
    #     FROM roll_map m
    #     WHERE m.start = {self.table_name}.start
    #         AND m.targ_ra    = {self.table_name}.targ_ra
    #         AND m.targ_dec   = {self.table_name}.targ_dec
    #     )
    #     WHERE EXISTS (
    #     SELECT 1
    #     FROM roll_map m
    #     WHERE m.start = {self.table_name}.start
    #         AND m.targ_ra    = {self.table_name}.targ_ra
    #         AND m.targ_dec   = {self.table_name}.targ_dec
    #     );
    #     """
    #     )
    #     self.conn.commit()
    #     self.cur.execute("DROP TABLE IF EXISTS roll_map")
    #     self.conn.commit()

    def _update_target_from_SOC(self):
        # This makes sure the database exists
        TargetDataBase()
        self.cur.execute(f"ATTACH DATABASE '{LEVEL0_DIR}/targets.db' AS targets")

        sql = """UPDATE pointings
                SET
                targ_ra = (
                    SELECT e.targ_ra
                    FROM targets e
                    WHERE e.targ_id = pointings.targ_id
                    AND e.created <= pointings.start
                    AND e.targ_ra IS NOT NULL
                    AND e.targ_ra = e.targ_ra             
                    ORDER BY e.created DESC
                    LIMIT 1
                ),
                targ_dec = (
                    SELECT e.targ_dec
                    FROM targets e
                    WHERE e.targ_id = pointings.targ_id
                    AND e.created <= pointings.start
                    AND e.targ_dec IS NOT NULL
                    AND e.targ_dec = e.targ_dec           
                    ORDER BY e.created DESC
                    LIMIT 1
                )
                WHERE
                (
                    pointings.targ_ra IS NULL OR pointings.targ_ra != pointings.targ_ra
                    OR pointings.targ_dec IS NULL OR pointings.targ_dec != pointings.targ_dec
                )
                AND EXISTS (
                    SELECT 1
                    FROM targets e
                    WHERE e.targ_id = pointings.targ_id
                    AND e.created <= pointings.start
                );"""

        self.conn.execute(sql)
        self.conn.commit()

        sql = """
            UPDATE pointings
            SET
            targ_ra = (
                SELECT t.targ_ra
                FROM targets.targets t
                WHERE t.targ_id = pointings.targ_id
                AND t.targ_ra IS NOT NULL
                AND t.targ_ra = t.targ_ra
                ORDER BY t.created ASC
                LIMIT 1
            ),
            targ_dec = (
                SELECT t.targ_dec
                FROM targets.targets t
                WHERE t.targ_id = pointings.targ_id
                AND t.targ_dec IS NOT NULL
                AND t.targ_dec = t.targ_dec
                ORDER BY t.created ASC
                LIMIT 1
            )
            WHERE
            (pointings.targ_ra IS NULL OR pointings.targ_ra != pointings.targ_ra
            OR pointings.targ_dec IS NULL OR pointings.targ_dec != pointings.targ_dec)
            AND EXISTS (
                SELECT 1
                FROM targets.targets t
                WHERE t.targ_id = pointings.targ_id
            );
            """

        self.conn.execute(sql)
        self.conn.commit()

        self.cur.execute("DETACH DATABASE targets;")

    def update_pointings(self):
        self._update_dpc_obs_id()
        self._update_target()
        self._update_start()
        self._update_target_from_SOC()
        # self._update_roll()
