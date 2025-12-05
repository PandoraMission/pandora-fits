# flake8: noqa W291
"""Tools for keeping a database of pandora files"""

import os
import sqlite3
import stat
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time

from . import DATA_DIR, DATABASE_DIR, logger


def update_filedatabase() -> None:
    """
    Creates and updates to the SQLite database file.
    """
    with FileDataBase() as db:
        db.crawl_and_add_parallel(DATA_DIR)
        db.update_pointings()


def delete_filedatabase() -> None:
    """
    Deletes the SQLite database file.

    Raises
    ------
    FileNotFoundError
        If the database file does not exist.
    """
    db_path = f"{DATABASE_DIR}/pointings.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.info(f"Database at {db_path} has been deleted.")
    else:
        raise FileNotFoundError(f"No database found at {db_path}.")


class FileDataBase(object):
    """Database for managing files."""

    def __init__(self):
        self.db_path = f"{DATABASE_DIR}/pointings.db"
        self.conn = sqlite3.connect(self.db_path)
        self.cur = self.conn.cursor()

        # 1. Create table (once)
        self.cur.execute(
            """
        CREATE TABLE IF NOT EXISTS pointings (
            filename TEXT PRIMARY KEY,
            dir TEXT,
            crsoftver TEXT,
            finetime INT,
            corstime INT,
            jd FLOAT,
            date STR,
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
        (filename, dir, crsoftver, finetime, corstime,
        jd, date, dpc_obs_id, start, instrmnt, roisizex,
        roisizey, roistrtx, roistrty, next, astrometry,
        targ_id, ra, dec, naxis1, naxis2, naxis3, naxis4, 
        badchecksum, baddatasum, filesize)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        self.conn.commit()

        os.chmod(
            self.db_path,
            stat.S_IRUSR
            | stat.S_IWUSR  # owner: read/write
            | stat.S_IRGRP
            | stat.S_IWGRP,  # group: read/write
        )

    def __repr__(self):
        return "Pandora FileDataBase"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()

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
                time = Time("2000-01-01 12:00:00", scale="tai") + timedelta(
                    seconds=hdr["CORSTIME"],
                    milliseconds=hdr["FINETIME"] / 1e6,
                )
                hdr1 = hdulist[1].header
                for key in ["FINETIME", "CORSTIME", "INSTRMNT"]:
                    if key not in hdr:
                        return
                return (
                    filename.split("/")[-1],
                    "/".join(filename.split("/")[:-1]),
                    hdr["CRSOFTV"],
                    hdr["FINETIME"],
                    hdr["CORSTIME"],
                    time.jd,
                    time.isot,
                    -1,
                    -1,
                    hdr["INSTRMNT"][0],
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

    def add_entry(self, values):
        self.cur.execute(
            self.update_str,
            values,
        )
        self.conn.commit()

    def add_entries(self, values):
        self.cur.executemany(
            self.update_str,
            values,
        )
        self.conn.commit()

    def crawl_and_add(self, root):
        for image_type in ["InfImg", "VisSci", "VisImg"]:
            # for path in Path(root).rglob(f"*{image_type}*.fits"):
            #     self.add_entry(self.get_entry(str(path)))
            self.add_entries(
                [
                    self.get_entry(str(path))
                    for path in Path(root).rglob(f"*{image_type}*.fits")
                    if not self.check_filename_in_database(str(path))
                ]
            )

    def crawl_and_add_parallel(self, root, max_workers=16):
        for image_type in ["InfImg", "VisSci", "VisImg"]:
            # for path in Path(root).rglob(f"*{image_type}*.fits"):
            #     self.add_entry(self.get_entry(str(path)))
            paths = [
                str(path)
                for path in Path(root).rglob(f"*{image_type}*.fits")
                if not self.check_filename_in_database(str(path))
            ]
            rows = []
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(self.get_entry, p) for p in paths]
                for fut in as_completed(futures):
                    row = fut.result()
                    if row is not None:
                        rows.append(row)
            self.add_entries(rows)

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

    def to_pandas(self, time_range=None, targ_id=None, dpc_obs_id=None):
        sql = "SELECT * FROM pointings"
        params = []

        where_clauses = []

        if time_range is not None:
            start, end = _process_time(time_range[0]), _process_time(time_range[1])
            start, end = np.sort([start, end])
            where_clauses.append("jd BETWEEN ? AND ?")
            params.extend([start, end])

        if targ_id is not None:
            where_clauses.append("targ_id = ?")
            params.append(targ_id)

        if dpc_obs_id is not None:
            where_clauses.append("dpc_obs_id = ?")
            params.append(dpc_obs_id)

        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        return pd.read_sql_query(sql, self.conn, params=params)


def _process_time(time):
    if isinstance(time, Time):
        time = time.jd
    elif not isinstance(time, float):
        try:
            time = Time(time).jd
        except ValueError:
            raise ValueError("`time_range` must be in JD.")
    return time
