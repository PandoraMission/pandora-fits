# flake8: noqa W291
"""Database tools for holding astrometry as measured on board."""

import os
import warnings
from datetime import timedelta
from pathlib import Path

from astropy.io import fits
from astropy.time import Time

from .. import DATA_DIR, __version__, LEVEL0_DIR
from ..utils import get_dpc_hashkey

import os
import warnings
from datetime import timedelta
from pathlib import Path

import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time

from .. import DATA_DIR, LEVEL0_DIR, __version__
from .mixins import DataBaseMixins
from .targets import TargetDataBase
import numpy as np
from astropy.coordinates import SkyCoord
from ..roll import get_roll


class AstrometryDataBase(DataBaseMixins):
    """Database for managing astrometry of Pandora"""

    table_name = "astrometry"
    db_path = f"{LEVEL0_DIR}/astrometry.db"
    _sql_key_dict = {
        "filename": "TEXT",
        "dir": "TEXT",
        "crsoftver": "TEXT",
        "pfsoftver": "TEXT",
        "start": "FLOAT",
        "end": "FLOAT",
        "jd": "FLOAT",
        "exptime": "FLOAT",
        "dpc_hash_key": "TEXT",
        "targ_id": "STR",
        "targ_ra": "FLOAT",
        "targ_dec": "FLOAT",
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
                        time.jd + ((temp_data[-1][0] / 1e3) / (24.0 * 60.0 * 60.0)),
                        time.jd + ((temp_data[idx][0] / 1e3) / (24.0 * 60.0 * 60.0)),
                        exptime,
                        hashkey,
                        hdr["TARG_ID"],
                        hdr["TARG_RA"],
                        hdr["TARG_DEC"],
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

    # def _update_roll(self):
    #     df = pd.read_sql_query(
    #         f"SELECT start, targ_id, targ_ra, targ_dec, targ_rll FROM {self.table_name} WHERE start = jd and targ_rll is NULL",
    #         self.conn,
    #     )
    #     with TargetDataBase() as db:
    #         for idx in df.index:
    #             query = f"SELECT visit_start, visit_end, targ_id, targ_ra, targ_dec, boresight_roll FROM {db.table_name} WHERE visit_start <= {df.iloc[idx]['start']} AND visit_end >= {df.iloc[idx]['start']} AND targ_id = '{df.iloc[idx]['targ_id']}'"
    #             targets = pd.read_sql_query(
    #                 query,
    #                 db.conn,
    #             )
    #             if len(targets) >= 1:
    #                 df.loc[idx, "targ_rll"] = targets.iloc[0]["boresight_roll"]
    #                 df.loc[idx, "targ_rll_type"] = "SOC"
    #             else:
    #                 df.loc[idx, "targ_rll"] = get_roll(
    #                     Time(df.loc[idx, "start"], format="jd"),
    #                     SkyCoord(
    #                         df.loc[idx, "targ_ra"], df.loc[idx, "targ_dec"], unit="deg"
    #                     ),
    #                 )[0].value
    #                 df.loc[idx, "targ_rll_type"] = "DPC PREDICT"

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
    #     for attr in ["targ_rll", "targ_rll_type"]:
    #         self.cur.execute(
    #             f"""
    #         UPDATE {self.table_name}
    #         SET {attr} = (
    #         SELECT m.{attr}
    #         FROM roll_map m
    #         WHERE m.start = {self.table_name}.start
    #             AND m.targ_ra    = {self.table_name}.targ_ra
    #             AND m.targ_dec   = {self.table_name}.targ_dec
    #         )
    #         WHERE EXISTS (
    #         SELECT 1
    #         FROM roll_map m
    #         WHERE m.start = {self.table_name}.start
    #             AND m.targ_ra    = {self.table_name}.targ_ra
    #             AND m.targ_dec   = {self.table_name}.targ_dec
    #         );
    #         """
    #         )
    #         self.conn.commit()

    #     self.cur.execute("DROP TABLE IF EXISTS roll_map")
    #     self.conn.commit()

    def update_pointings(self):
        # self._update_roll()
        return
