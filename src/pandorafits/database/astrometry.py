# flake8: noqa W291
"""Database tools for holding astrometry as measured on board."""

import os
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
from ..utils import get_dpc_hashkey
from .mixins import DataBaseMixins


class AstrometryDataBase(DataBaseMixins):
    """Database for managing astrometry of Pandora"""

    table_name = "astrometry"
    db_path = f"{LEVEL0_DIR}/astrometry.db"
    _sql_key_dict = {
        "filename": "TEXT",
        "dir": "TEXT",
        "crsoftver": "TEXT",
        "pfsoftver": "TEXT",
        "jd0": "FLOAT",
        "jd": "FLOAT",
        "exptime": "FLOAT",
        # "dpc_obs_id": "INT",
        "dpchashkey": "TEXT",
        "targ_id": "STR",
        "targ_ra": "FLOAT",
        "targ_dec": "FLOAT",
        "targ_rll": "FLOAT",
        "has_astrometry": "INT",
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
                hashkey = get_dpc_hashkey(
                    hdr["TARG_ID"],
                    hdr["TARG_RA"],
                    hdr["TARG_DEC"],
                )

                if ("ASTROMETRY" in hdulist) and ("TEMP_TIME" in hdulist):
                    astrometry_data = hdulist["ASTROMETRY"].data
                    temp_data = hdulist["TEMP_TIME"].data
                    has_astrometry = True
                else:
                    astrometry_data = np.asarray([[0.0, 0.0, 40.0]])
                    temp_data = np.asarray([0.0, -99.0])
                    has_astrometry = False
                return [
                    (
                        filename.split("/")[-1],
                        "/".join(filename.split("/")[:-1]),
                        hdr["CRSOFTV"],
                        __version__,
                        time.jd,
                        time.jd + ((temp_data[idx][0] / 1e3) / (24.0 * 60.0 * 60.0)),
                        exptime,
                        hashkey,
                        hdr["TARG_ID"],
                        hdr["TARG_RA"],
                        hdr["TARG_DEC"],
                        -1,
                        int(has_astrometry),
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

    # def _update_dpc_obs_id(self):
    #     sql = f"""
    #     WITH changes AS (
    #     SELECT
    #         targ_id,
    #         CASE
    #         WHEN targ_id = LAG(targ_id) OVER (ORDER BY jd, targ_id)
    #         THEN 0          -- same target as previous row → same visit
    #         ELSE 1          -- target changed (or first row) → new visit
    #         END AS is_new_visit
    #     FROM {self.table_name}
    #     ),
    #     visits AS (
    #     SELECT
    #         targ_id,
    #         SUM(is_new_visit) OVER (ORDER BY jd, targ_id) AS dpc_obs_id
    #     FROM changes
    #     )
    #     UPDATE {self.table_name}
    #     SET dpc_obs_id = (
    #     SELECT dpc_obs_id FROM visits WHERE visits.targ_id = {self.table_name}.targ_id
    #     );

    #     """
    #     self.conn.execute(sql)

    # def _update_start(self):
    #     sql = f"""
    #     WITH filled AS (
    #     SELECT
    #         targ_id,
    #         MIN(jd) OVER (
    #         PARTITION BY dpc_obs_id
    #         ORDER BY jd
    #         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    #         ) AS start_filled
    #     FROM {self.table_name}
    #     )
    #     UPDATE {self.table_name}
    #     SET start = (SELECT start_filled FROM filled WHERE filled.targ_id = {self.table_name}.targ_id)
    #     """
    #     self.conn.execute(sql)

    def _update_roll(self):
        df = pd.read_sql_query(
            f"SELECT jd0, targ_ra, targ_dec FROM {self.table_name} WHERE jd0 = jd",
            self.conn,
        )
        df["targ_rll"] = [
            get_roll(Time(jd0, format="jd"), SkyCoord(ra, dec, unit="deg"))[0].value
            for jd0, ra, dec in df.values
        ]

        # 1) write df to a temp table
        df.to_sql("roll_map", self.conn, if_exists="replace", index=False)

        # 2) (optional but strongly recommended) index the join keys in both tables
        self.cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_main_keys ON {self.table_name}(jd0, targ_ra, targ_dec)"
        )
        self.cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_map_keys  ON roll_map(jd0, targ_ra, targ_dec)"
        )
        self.conn.commit()

        # 3) update matching rows (fills duplicates in main table too)
        self.cur.execute(
            f"""
        UPDATE {self.table_name}
        SET targ_rll = (
        SELECT m.targ_rll
        FROM roll_map m
        WHERE m.jd0 = {self.table_name}.jd0
            AND m.targ_ra    = {self.table_name}.targ_ra
            AND m.targ_dec   = {self.table_name}.targ_dec
        )
        WHERE EXISTS (
        SELECT 1
        FROM roll_map m
        WHERE m.jd0 = {self.table_name}.jd0
            AND m.targ_ra    = {self.table_name}.targ_ra
            AND m.targ_dec   = {self.table_name}.targ_dec
        );
        """
        )
        self.conn.commit()
        self.cur.execute("DROP TABLE IF EXISTS roll_map")
        self.conn.commit()

    def update_pointings(self):
        # self._update_dpc_obs_id()
        # self._update_start()
        self._update_roll()
