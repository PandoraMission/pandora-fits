# flake8: noqa W291
"""Database tools for MOC files database"""

# Standard library
import os
import sqlite3
import stat
import warnings
from datetime import timedelta
from pathlib import Path

try:
    # Third-party
    import fitsio
except ImportError:  # pragma: no cover - exercised only when fitsio is absent
    fitsio = None
# Third-party
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .. import DATA_DIR, LEVEL0_DIR, __version__, logger
from ..roll import get_roll
from ..utils import convert_time, get_dpc_hashkey, get_exposure_time
from . import DPC_KEYS
from .astrometry import AstrometryDataBase
from .mixins import DataBaseMixins
from .targets import TargetDataBase


def get_entry(filename, checksums=False):
    if fitsio is None:
        raise ModuleNotFoundError(
            "fitsio is required for database file parsing but is not available on this platform."
        )

    filesize = os.path.getsize(filename) / (1024 * 1024)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # capture all warnings

        with fitsio.FITS(filename) as hdulist:
            if len(hdulist) <= 1:
                return

            badchecksum, baddatasum = 0, 0
            if checksums:
                try:
                    hdulist[1].verify_checksum()
                except ValueError as e:
                    if "data" in str(e):
                        baddatasum += 1
                    elif "hdu" in str(e):
                        badchecksum += 1
                    else:
                        raise e

            hdr = hdulist[0].read_header()
            time = convert_time(hdr["CORSTIME"], hdr["FINETIME"])
            hdr1 = hdulist[1].read_header()
            exptime = get_exposure_time(hdr).value

            # elif "EXPTIMEU" in hdr:
            #     if "NAXIS3" in hdr1:
            #         frame_time = (
            #             hdr["EXPTIMEU"] * hdr["FRMSCLCT"] / hdr1["NAXIS3"]
            #         ) / 1.0e6
            #     else:
            #         frame_time = (hdr["EXPTIMEU"] * hdr["FRMSCLCT"]) / 1.0e6
            # elif "EXPTIME" in hdr:
            #     if "NAXIS3" in hdr1:
            #         frame_time = (
            #             hdr["EXPTIME"] * hdr["FRMSCLCT"] / hdr1["NAXIS3"]
            #         ) / 1.0e6
            #     else:
            #         frame_time = (hdr["EXPTIME"] * hdr["FRMSCLCT"]) / 1.0e6

            # nframes = hdr1[f"NAXIS{hdr1['NAXIS']}"]
            # exptime = nframes * frame_time
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
                np.round(time.jd, 6),
                time.isot,
                exptime,
                None,
                np.round(time.jd, 6),
                hdr["INSTRMNT"],
                hdr["ROISIZEX"],
                hdr["ROISIZEY"],
                hdr["ROISTRTX"],
                hdr["ROISTRTY"],
                len(hdulist),
                "ASTROMETRY"
                in np.asarray([hdu.get_extname() for hdu in hdulist]),
                hdr["TARG_ID"] if "TARG_ID" in hdr else None,
                hdr["TARG_RA"] if "TARG_RA" in hdr else None,
                hdr["TARG_DEC"] if "TARG_DEC" in hdr else None,
                None,
                None,
                None,
                None,
                None,
                hdr1["NAXIS1"] if "NAXIS1" in hdr1 else None,
                hdr1["NAXIS2"] if "NAXIS2" in hdr1 else None,
                hdr1["NAXIS3"] if "NAXIS3" in hdr1 else None,
                hdr1["NAXIS4"] if "NAXIS4" in hdr1 else None,
                badchecksum,
                baddatasum,
                filesize,
            )


class Level0DataBase(DataBaseMixins):
    """Database for managing files that have been delivered by MOC."""

    table_name = "pointings"
    db_path = f"{LEVEL0_DIR}/level0.db"
    level = 0
    _sql_key_dict = DPC_KEYS

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

    # def get_entry(self, filename):
    #     filesize = os.path.getsize(filename) / (1024 * 1024)
    #     with warnings.catch_warnings(record=True) as w:
    #         warnings.simplefilter("always")  # capture all warnings

    #         with fits.open(filename, lazy_load_hdus=True) as hdulist:
    #             if len(hdulist) <= 1:
    #                 return
    #             badchecksum = len(
    #                 [warn for warn in w if "Checksum" in str(warn.message)]
    #             )
    #             baddatasum = len(
    #                 [warn for warn in w if "Datasum" in str(warn.message)]
    #             )

    #             hdr = hdulist[0].header
    #             time = (
    #                 Time("2000-01-01 12:00:00", scale="tai")
    #                 + timedelta(
    #                     seconds=hdr["CORSTIME"],
    #                     milliseconds=hdr["FINETIME"] / 1e6,
    #                 )
    #             ).utc
    #             hdr1 = hdulist[1].header

    #             if "FRMTIME" in hdr:
    #                 frame_time = hdr["FRMTIME"] / 1000
    #             elif "EXPTIMEU" in hdr:
    #                 frame_time = (
    #                     hdr["EXPTIMEU"] * hdr["FRMSCLCT"] / hdr1["NAXIS3"]
    #                 ) / 1.0e6
    #             elif "EXPTIME" in hdr:
    #                 frame_time = (
    #                     hdr["EXPTIME"] * hdr["FRMSCLCT"] / hdr1["NAXIS3"]
    #                 ) / 1.0e6

    #             nframes = hdr1[f"NAXIS{hdr1['NAXIS']}"]
    #             exptime = nframes * frame_time
    #             for key in ["FINETIME", "CORSTIME", "INSTRMNT"]:
    #                 if key not in hdr:
    #                     return
    #             return (
    #                 filename.split("/")[-1],
    #                 filename.split("/")[-1],
    #                 "/".join(filename.split("/")[:-1]),
    #                 "/".join(filename.split("/")[:-1]),
    #                 hdr["CRSOFTV"],
    #                 __version__,
    #                 hdr["FINETIME"],
    #                 hdr["CORSTIME"],
    #                 time.jd,
    #                 time.isot,
    #                 exptime,
    #                 None,
    #                 None,
    #                 None,
    #                 hdr["INSTRMNT"],
    #                 hdr["ROISIZEX"],
    #                 hdr["ROISIZEY"],
    #                 hdr["ROISTRTX"],
    #                 hdr["ROISTRTY"],
    #                 len(hdulist),
    #                 "ASTROMETRY"
    #                 in np.asarray([hdu.header["extname"] for hdu in hdulist]),
    #                 hdr["TARG_ID"] if "TARG_ID" in hdr else None,
    #                 hdr["TARG_RA"] if "TARG_RA" in hdr else None,
    #                 hdr["TARG_DEC"] if "TARG_DEC" in hdr else None,
    #                 None,
    #                 None,
    #                 None,
    #                 None,
    #                 None,
    #                 hdr1["NAXIS1"] if "NAXIS1" in hdr1 else None,
    #                 hdr1["NAXIS2"] if "NAXIS2" in hdr1 else None,
    #                 hdr1["NAXIS3"] if "NAXIS3" in hdr1 else None,
    #                 hdr1["NAXIS4"] if "NAXIS4" in hdr1 else None,
    #                 badchecksum,
    #                 baddatasum,
    #                 filesize,
    #             )

    def crawl_and_add(self, match_str=""):
        root = DATA_DIR
        for image_type in ["InfImg", "VisSci", "VisImg"]:
            # for path in Path(root).rglob(f"*{image_type}*.fits"):
            #     self.add_entry(self.get_entry(str(path)))
            paths = np.asarray(
                [
                    str(path)
                    for path in Path(root).rglob(
                        f"*{match_str}*{image_type}*.fits"
                        if match_str != ""
                        else f"*{image_type}*.fits"
                    )
                    if not self.check_filename_in_database(str(path))
                ]
            )
            path_idxs = np.unique(
                [p.split("/")[-1] for p in paths], return_index=True
            )[1]
            paths = np.sort(paths[path_idxs])
            rows = []
            with logging_redirect_tqdm():
                for path in tqdm(paths, desc=f"{image_type} files"):
                    values = get_entry(path)
                    if values is not None:
                        rows.append(values)
            self.add_entries(rows)
        self.update_pointings()

    # def _update_dpc_seq_id(self):
    #     sql = f"""
    #     WITH changes AS (
    #     SELECT
    #         targ_id,
    #         CASE
    #         WHEN targ_id = LAG(targ_id) OVER (ORDER BY jd, targ_id)
    #         THEN 0          -- same target as previous row → same sequence
    #         ELSE 1          -- target changed (or first row) → new sequence
    #         END AS is_new_sequence
    #     FROM {self.table_name}
    #     ),
    #     sequences AS (
    #     SELECT
    #         targ_id,
    #         SUM(is_new_sequence) OVER (ORDER BY jd, targ_id) AS dpc_seq_id
    #     FROM changes
    #     )
    #     UPDATE {self.table_name}
    #     SET dpc_seq_id = (
    #     SELECT dpc_seq_id FROM sequences WHERE sequences.targ_id = {self.table_name}.targ_id
    #     );

    #     """
    #     self.conn.execute(sql)

    def _update_target(self):
        sql = f"""
        UPDATE {self.table_name} AS n
        SET
            targ_ra = (
                SELECT v.targ_ra
                FROM {self.table_name} AS v
                WHERE v.instrmnt = 'VISDA'
                AND v.targ_id = n.targ_id
                AND v.jd BETWEEN n.jd - 0.001 AND n.jd + 0.001
                AND v.targ_ra IS NOT NULL
                AND v.targ_dec IS NOT NULL
                ORDER BY v.jd ASC
                LIMIT 1
            ),
            targ_dec = (
                SELECT v.targ_dec
                FROM {self.table_name} AS v
                WHERE v.instrmnt = 'VISDA'
                AND v.targ_id = n.targ_id
                AND v.jd BETWEEN n.jd - 0.001 AND n.jd + 0.001
                AND v.targ_ra IS NOT NULL
                AND v.targ_dec IS NOT NULL
                ORDER BY v.jd ASC
                LIMIT 1
            )
        WHERE n.instrmnt = 'NIRDA'
        AND (
            n.targ_ra IS NULL OR n.targ_ra != n.targ_ra
            OR n.targ_dec IS NULL OR n.targ_dec != n.targ_dec
        );
        """
        logger.info("Filling in nan targets with recent VISDA data.")
        self.conn.execute(sql)
        self.conn.commit()

    def _update_start(self):
        # sql = f"""
        # UPDATE {self.table_name} AS p1
        # SET start = MIN(p1.jd, p2.jd)
        # FROM {self.table_name} AS p2
        # WHERE p1.instrmnt != p2.instrmnt
        # AND p1.targ_id = p2.targ_id
        # AND p2.jd BETWEEN p1.jd - 0.001 AND p1.jd + 0.001;
        # """

        sql = f"""
            WITH pairs AS (
                SELECT
                    v.rowid AS visda_rowid,
                    n.rowid AS nirda_rowid,
                    v.jd AS visda_jd
                FROM {self.table_name} AS v
                JOIN {self.table_name} AS n
                ON v.targ_id = n.targ_id
                AND n.jd BETWEEN v.jd - 0.001 AND v.jd + 0.001
                WHERE v.instrmnt = 'VISDA'
                AND n.instrmnt = 'NIRDA'
            )
            UPDATE {self.table_name}
            SET start = (
                SELECT visda_jd
                FROM pairs
                WHERE {self.table_name}.rowid IN (pairs.visda_rowid, pairs.nirda_rowid)
                LIMIT 1
            )
            WHERE rowid IN (
                SELECT visda_rowid FROM pairs
                UNION
                SELECT nirda_rowid FROM pairs
            );
            """
        self.conn.execute(sql)
        self.conn.commit()

    def _update_pointing(self):
        self.cur.execute(
            f"ATTACH DATABASE '{LEVEL0_DIR}/astrometry.db' AS astrometry"
        )

        # If there are nans in the targ_ra, targ_dec, fill them in with the most recent values from the SOC before the data.
        sql = f"""WITH matched AS (
        SELECT
            p.rowid AS p_rowid,
            AVG(a.ra) AS mean_ra,
                    AVG(a.dec) AS mean_dec,
                    AVG(a.roll) AS mean_roll
                FROM pointings AS p
                JOIN astrometry AS a
                ON a.start = p.start
                GROUP BY p.rowid
            )
            UPDATE pointings
            SET
                ra = (SELECT mean_ra FROM matched WHERE matched.p_rowid = pointings.rowid),
                dec = (SELECT mean_dec FROM matched WHERE matched.p_rowid = pointings.rowid),
                roll = (SELECT mean_roll FROM matched WHERE matched.p_rowid = pointings.rowid)
            WHERE rowid IN (
                SELECT p_rowid FROM matched
            );"""

        self.conn.execute(sql)

        # logger.info("Filled in on board pointing result for exact matches.")
        self.cur.execute("DETACH DATABASE astrometry;")
        self.conn.commit()

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

    # def _update_target_from_SOC(self):
    #     # This makes sure the database exists
    #     TargetDataBase()
    #     self.cur.execute(f"ATTACH DATABASE '{LEVEL0_DIR}/targets.db' AS targets")

    #     # If there are nans in the targ_ra, targ_dec, fill them in with the most recent values from the SOC before the data.
    #     sql = """UPDATE pointings
    #             SET
    #             targ_ra = (
    #                 SELECT e.targ_ra
    #                 FROM targets e
    #                 WHERE e.targ_id = pointings.targ_id
    #                 AND e.created <= pointings.start
    #                 db_path = os.path.join(LEVEL0_DIR, "level0.db")
    #                 AND e.targ_ra = e.targ_ra
    #                 ORDER BY e.created DESC
    #                 LIMIT 1
    #             ),
    #             targ_dec = (
    #                 SELECT e.targ_dec
    #                         basename = os.path.basename(filename)
    #                         dirname = os.path.dirname(filename)
    #                 FROM targets e
    #                             basename,
    #                             basename,
    #                             dirname,
    #                             dirname,
    #                 ORDER BY e.created DESC
    #                 LIMIT 1
    #             )
    #             WHERE
    #             (
    #                 pointings.targ_ra IS NULL OR pointings.targ_ra != pointings.targ_ra
    #                 OR pointings.targ_dec IS NULL OR pointings.targ_dec != pointings.targ_dec
    #             )
    #             AND EXISTS (
    #                 SELECT 1
    #                 FROM targets e
    #                 WHERE e.targ_id = pointings.targ_id
    #                 AND e.created <= pointings.start
    #             );"""

    #     logger.info("Filling in nan targets with recent SOC data.")

    #     self.conn.execute(sql)
    #     self.conn.commit()

    #     # If that doesn't work select any matching target ID
    #     sql = """
    #         UPDATE pointings
    #         SET
    #         targ_ra = (
    #             SELECT t.targ_ra
    #             FROM targets.targets t
    #             WHERE t.targ_id = pointings.targ_id
    #             AND t.targ_ra IS NOT NULL
    #             AND t.targ_ra = t.targ_ra
    #             ORDER BY t.created ASC
    #             LIMIT 1
    #         ),
    #         targ_dec = (
    #             SELECT t.targ_dec
    #             FROM targets.targets t
    #                         f"ATTACH DATABASE '{os.path.join(LEVEL0_DIR, 'targets.db')}' AS targets"
    #             AND t.targ_dec IS NOT NULL
    #             AND t.targ_dec = t.targ_dec
    #                         f"ATTACH DATABASE '{os.path.join(LEVEL0_DIR, 'astrometry.db')}' AS astrometry"
    #             LIMIT 1
    #         )
    #         WHERE
    #         (pointings.targ_ra IS NULL OR pointings.targ_ra != pointings.targ_ra
    #         OR pointings.targ_dec IS NULL OR pointings.targ_dec != pointings.targ_dec)
    #         AND EXISTS (
    #             SELECT 1
    #             FROM targets.targets t
    #             WHERE t.targ_id = pointings.targ_id
    #         );
    #         """

    #     self.conn.execute(sql)
    #     self.conn.commit()
    #     logger.info("Filling in remaining nan targets with any SOC data.")

    #     self.cur.execute("DETACH DATABASE targets;")

    #     def _update_roll_from_SOC(self):
    #         # This makes sure the database exists
    #         TargetDataBase()
    #         self.cur.execute(f"ATTACH DATABASE '{LEVEL0_DIR}/targets.db' AS targets")

    #         sql = """UPDATE pointings
    # SET
    #   targ_rll = (
    #     SELECT t.boresight_roll
    #     FROM targets.targets AS t
    #     WHERE t.targ_id = pointings.targ_id
    #       AND pointings.start BETWEEN t.visit_start AND t.visit_end
    #     LIMIT 1
    #   ),
    #   targ_rll_type = 'SOC'
    # WHERE targ_rll IS NULL OR targ_rll != targ_rll
    #   AND EXISTS (
    #     SELECT 1
    #     FROM targets.targets AS t
    #     WHERE t.targ_id = pointings.targ_id
    #       AND pointings.start BETWEEN t.visit_start AND t.visit_end
    #   );"""

    # self.conn.execute(sql)
    # self.conn.commit()
    # logger.info("Filling in targ_rll with SOC data.")
    # self.cur.execute("DETACH DATABASE targets;")

    # def _update_roll_from_SOC(self):
    #     df = pd.read_sql_query(
    #         f"SELECT start, targ_id, targ_ra, targ_dec, targ_rll FROM {self.table_name} WHERE start = jd and (targ_rll is NULL OR targ_rll != targ_rll)",
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
    #                         df.loc[idx, "targ_ra"],
    #                         df.loc[idx, "targ_dec"],
    #                         unit="deg",
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

    # def _update_pointing_from_payload(self):
    #     # This makes sure the database exists
    #     AstrometryDataBase()
    #     self.cur.execute(f"ATTACH DATABASE '{LEVEL0_DIR}/astrometry.db' AS astrometry")

    #     # If there are nans in the targ_ra, targ_dec, fill them in with the most recent values from the SOC before the data.
    #     for attr in ["ra", "dec", "roll"]:
    #         sql = f"""UPDATE pointings AS p
    #                 SET {attr} = (
    #                     SELECT AVG(a.{attr})
    #                     FROM astrometry.astrometry AS a
    #                     WHERE a.jd >= (p.start - 30/86400)
    #                     AND a.jd <  p.start + (p.exptime / 86400.0) + (30/86400)
    #                 );"""

    #         self.conn.execute(sql)
    #         self.conn.commit()

    #     logger.info("Filled in on board pointing result for exact matches.")
    #     self.cur.execute("DETACH DATABASE astrometry;")

    def _update_dpc_hash_key(self):
        self.cur = self.conn.cursor()

        self.cur.execute(
            """
            SELECT DISTINCT targ_id, targ_ra, targ_dec
            FROM pointings
            WHERE dpc_hash_key IS NULL
        """
        )

        rows = self.cur.fetchall()
        hash_rows = []
        for targ_id, ra, dec in rows:
            hk = get_dpc_hashkey(targ_id, ra, dec)
            hash_rows.append((hk, targ_id, ra, dec))
        self.cur.executemany(
            """
            UPDATE pointings
            SET dpc_hash_key = ?
            WHERE targ_id = ?
            AND targ_ra IS ?
            AND targ_dec IS ?
        """,
            hash_rows,
        )
        self.conn.commit()
        logger.info("Filled in DPC hash keys.")

    def update_pointings(self):
        self._update_target()
        self._update_start()
        self._update_pointing()
        self._update_dpc_hash_key()
        # self._update_pointing_from_payload()
        # self._update_dpc_seq_id()
        # self._update_target_from_SOC()
        # self._update_roll_from_SOC()
        # self._update_pointing_from_payload()
        # self._update_dpc_hash_key()
        # self._update_roll()
