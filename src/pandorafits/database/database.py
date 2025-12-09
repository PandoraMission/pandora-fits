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

from . import DATA_DIR, LEVEL0_DIR, logger, LEVEL1_DIR, __version__, CRSOFTVER
from .visda import VISDAFFILevel0HDUList, VISDALevel0HDUList
from .nirda import NIRDALevel0HDUList
from .utils import get_dpc_hashkey


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
    db_path = f"{LEVEL0_DIR}/pointings.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.warning(f"Database at {db_path} has been deleted.")
    else:
        logger.warning(f"Tried to delete FileDataBase. No database found at {db_path}.")


def delete_level1database() -> None:
    """
    Deletes the SQLite database file.

    Raises
    ------
    FileNotFoundError
        If the database file does not exist.
    """
    db_path = f"{LEVEL1_DIR}/level1.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.warning(f"Database at {db_path} has been deleted.")
    else:
        logger.warning(
            f"Tried to delete Level1DataBase. No database found at {db_path}."
        )


def clean_level1database(crsoftver=CRSOFTVER, pfsoftver=__version__):
    """Be warned do NOT run this on multiple nodes, run on one node!"""
    logger.info("Running `clean_level1database`")
    with Level1DataBase() as l1db:
        for file in l1db.files_to_process(crsoftver=crsoftver):
            l1db.check_file_in_database(
                file, crsoftver=crsoftver, pfsoftver=pfsoftver, remove_bad_entries=True
            )
    logger.info("Finished `clean_level1database`")


class DataBaseMixins:
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


class FileDataBase(DataBaseMixins):
    """Database for managing files that have been delivered by MOC."""

    def __init__(self):
        self.db_path = f"{LEVEL0_DIR}/pointings.db"
        self.conn = sqlite3.connect(self.db_path)
        self.cur = self.conn.cursor()

        self.cur.execute(
            """
        CREATE TABLE IF NOT EXISTS pointings (
            filename TEXT PRIMARY KEY,
            dir TEXT,
            crsoftver TEXT,
            pfsoftver TEXT,
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
        (filename, dir, crsoftver, pfsoftver, finetime, corstime,
        jd, date, dpc_obs_id, start, instrmnt, roisizex,
        roisizey, roistrtx, roistrty, next, astrometry,
        targ_id, ra, dec, naxis1, naxis2, naxis3, naxis4, 
        badchecksum, baddatasum, filesize)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
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
                time = (
                    Time("2000-01-01 12:00:00", scale="tai")
                    + timedelta(
                        seconds=hdr["CORSTIME"],
                        milliseconds=hdr["FINETIME"] / 1e6,
                    )
                ).utc
                hdr1 = hdulist[1].header
                for key in ["FINETIME", "CORSTIME", "INSTRMNT"]:
                    if key not in hdr:
                        return
                return (
                    filename.split("/")[-1],
                    "/".join(filename.split("/")[:-1]),
                    hdr["CRSOFTV"],
                    __version__,
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


class Level1DataBase(DataBaseMixins):
    """Database for managing Level 1 files."""

    def __init__(self):
        self.db_path = f"{LEVEL1_DIR}/level1.db"
        self.conn = sqlite3.connect(self.db_path)
        self.cur = self.conn.cursor()
        self.cur.execute(
            """
        CREATE TABLE IF NOT EXISTS pointings (
            filename TEXT PRIMARY KEY,
            dir TEXT,
            crsoftver TEXT,
            pfsoftver TEXT,
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
        (filename, dir, crsoftver, pfsoftver, finetime, corstime,
        jd, date, dpc_obs_id, start, instrmnt, roisizex,
        roisizey, roistrtx, roistrty, next, astrometry,
        targ_id, ra, dec, naxis1, naxis2, naxis3, naxis4, 
        badchecksum, baddatasum, filesize)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        self.conn.commit()

        os.chmod(
            self.db_path,
            stat.S_IRUSR
            | stat.S_IWUSR  # owner: read/write
            | stat.S_IRGRP
            | stat.S_IWGRP,  # group: read/write
        )
        # Add the level0 database in
        self.cur.execute(f"ATTACH DATABASE '{LEVEL0_DIR}/pointings.db' AS level0")

    def __repr__(self):
        return "Pandora Level1DataBase"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()

    def _is_bad_value(
        self, fname, expected_entry, entry, name, level, remove_bad_entries=True
    ):
        if expected_entry != entry:
            logger.warning(
                f"File {fname} exists in Level {level} database with `{name}` set to `{entry}` requested value was `{expected_entry}`."
            )
            if remove_bad_entries:
                if level == 0:
                    self.cur.execute(
                        "DELETE FROM level0.pointings WHERE filename=?", (fname,)
                    )
                if level == 1:
                    self.cur.execute("DELETE FROM pointings WHERE filename=?", (fname,))
                self.conn.commit()
                logger.warning(f"File {fname} removed from Level {level} database.")
            return True
        return False

    def _is_bad_row(
        self,
        filename,
        crsoftver=CRSOFTVER,
        pfsoftver=__version__,
        badchecksum=0,
        baddatasum=0,
        level=0,
        remove_bad_entries=False,
    ):
        fname = filename.split("/")[-1] if "/" in filename else filename
        self.cur.execute(
            f"SELECT crsoftver, pfsoftver, badchecksum, baddatasum FROM {'level0.' if level == 0 else ''}pointings WHERE filename=?",
            (fname,),
        )
        row = self.cur.fetchone()

        if row is None:
            return False
        (
            database_crsoftver,
            database_pfsoftver,
            database_badchecksum,
            database_baddatasum,
        ) = row
        is_bad_row = False
        # Log any discrepancies
        if crsoftver is not None:
            is_bad_row |= self._is_bad_value(
                fname=fname,
                expected_entry=crsoftver,
                entry=database_crsoftver,
                name="CRSOFTV",
                level=0,
                remove_bad_entries=remove_bad_entries,
            )
        if pfsoftver is not None:
            if database_pfsoftver is not None:
                is_bad_row |= self._is_bad_value(
                    fname=fname,
                    expected_entry=pfsoftver,
                    entry=database_pfsoftver,
                    name="PFSOFTV",
                    level=0,
                    remove_bad_entries=remove_bad_entries,
                )
        if badchecksum is not None:
            is_bad_row |= self._is_bad_value(
                fname=fname,
                expected_entry=badchecksum,
                entry=database_badchecksum,
                name="badchecksum",
                level=0,
                remove_bad_entries=remove_bad_entries,
            )
        if baddatasum is not None:
            is_bad_row |= self._is_bad_value(
                fname=fname,
                expected_entry=baddatasum,
                entry=database_baddatasum,
                name="baddatasum",
                level=0,
                remove_bad_entries=remove_bad_entries,
            )
        return is_bad_row

    def check_filename_in_database(self, filename, level=0):
        fname = filename.split("/")[-1] if "/" in filename else filename
        self.cur.execute(
            f"SELECT crsoftver, badchecksum, baddatasum FROM {'level0.' if level == 0 else ''}pointings WHERE filename=?",
            (fname,),
        )
        return self.cur.fetchone() is not None

    # def _check_entry(
    #     self, fname, expected_entry, entry, name, level, remove_bad_entries=True
    # ):
    #     if expected_entry != entry:
    #         logger.warning(
    #             f"File {fname} exists in Level {level} database with `{name}` set to `{entry}` requested value was `{expected_entry}`."
    #         )
    #         if remove_bad_entries:
    #             if level == 0:
    #                 self.cur.execute(
    #                     "DELETE FROM level0.pointings WHERE filename=?", (fname,)
    #                 )
    #             if level == 1:
    #                 self.cur.execute("DELETE FROM pointings WHERE filename=?", (fname,))
    #             self.conn.commit()
    #             logger.warning(f"File {fname} removed from Level {level} database.")

    # def check_file_in_level0_database(
    #     self,
    #     filename,
    #     crsoftver=None,
    #     badchecksum=0,
    #     baddatasum=0,
    #     remove_bad_entries=False,
    # ):
    #     fname = filename.split("/")[-1] if "/" in filename else filename
    #     self.cur.execute(
    #         "SELECT crsoftver, badchecksum, baddatasum FROM level0.pointings WHERE filename=?",
    #         (fname,),
    #     )
    #     row = self.cur.fetchone()

    #     if row is None:
    #         return False

    #     database_crsoftver, database_badchecksum, database_baddatasum = row

    #     # Log any discrepancies
    #     if crsoftver is not None:
    #         self._check_entry(
    #             fname=fname,
    #             expected_entry=crsoftver,
    #             entry=database_crsoftver,
    #             name="CRSOFTV",
    #             level=0,
    #             remove_bad_entries=remove_bad_entries,
    #         )
    #     if badchecksum is not None:
    #         self._check_entry(
    #             fname=fname,
    #             expected_entry=badchecksum,
    #             entry=database_badchecksum,
    #             name="badchecksum",
    #             level=0,
    #             remove_bad_entries=remove_bad_entries,
    #         )
    #     if baddatasum is not None:
    #         self._check_entry(
    #             fname=fname,
    #             expected_entry=baddatasum,
    #             entry=database_baddatasum,
    #             name="baddatasum",
    #             level=0,
    #             remove_bad_entries=remove_bad_entries,
    #         )
    #     if remove_bad_entries:
    #         return False
    #     return True

    # def check_file_in_database(
    #     self,
    #     filename,
    #     crsoftver=None,
    #     pfsoftver=None,
    #     badchecksum=0,
    #     baddatasum=0,
    #     remove_bad_entries=False,
    # ):
    #     fname = filename.split("/")[-1] if "/" in filename else filename
    #     self.cur.execute(
    #         "SELECT crsoftver, pfsoftver, badchecksum, baddatasum FROM pointings WHERE filename=?",
    #         (fname,),
    #     )
    #     row = self.cur.fetchone()

    #     if row is None:
    #         return False
    #     (
    #         database_crsoftver,
    #         database_pfsoftver,
    #         database_badchecksum,
    #         database_baddatasum,
    #     ) = row
    #     # Log any discrepancies
    #     if crsoftver is not None:
    #         self._check_entry(
    #             fname=fname,
    #             expected_entry=crsoftver,
    #             entry=database_crsoftver,
    #             name="CRSOFTV",
    #             level=1,
    #             remove_bad_entries=remove_bad_entries,
    #         )
    #     if pfsoftver is not None:
    #         self._check_entry(
    #             fname=fname,
    #             expected_entry=pfsoftver,
    #             entry=database_pfsoftver,
    #             name="PFSOFTV",
    #             level=1,
    #             remove_bad_entries=remove_bad_entries,
    #         )
    #     if badchecksum is not None:
    #         self._check_entry(
    #             fname=fname,
    #             expected_entry=badchecksum,
    #             entry=database_badchecksum,
    #             name="badchecksum",
    #             level=1,
    #             remove_bad_entries=remove_bad_entries,
    #         )
    #     if baddatasum is not None:
    #         self._check_entry(
    #             fname=fname,
    #             expected_entry=baddatasum,
    #             entry=database_baddatasum,
    #             name="baddatasum",
    #             level=1,
    #             remove_bad_entries=remove_bad_entries,
    #         )
    #     if remove_bad_entries:
    #         return False
    #     return True

    def files_to_process(self, crsoftver=CRSOFTVER):
        cur = self.conn.cursor()
        cur.execute("SELECT filename, dir FROM level0.pointings")

        row = cur.fetchone()
        while row is not None:
            if (not self.check_filename_in_database(row[0], level=1)) & (
                not self._is_bad_row(row[0], crsoftver=crsoftver, level=0)
            ):
                yield (f"{row[1]}/{row[0]}")
            row = cur.fetchone()

    def get_entry(self, filename):
        fname = filename.split("/")[-1] if "/" in filename else filename
        self.cur.execute(
            "SELECT * FROM level0.pointings WHERE filename=?",
            (fname,),
        )
        row = self.cur.fetchone()
        return (*row[:3], __version__, *row[3:])

    def get_level1_filename(self, filename):
        fname = filename.split("/")[-1] if "/" in filename else filename
        self.cur.execute(
            "SELECT start, targ_id, ra, dec FROM level0.pointings WHERE filename=?",
            (fname,),
        )
        row = self.cur.fetchone()
        if row is None:
            return None
        t = Time(row[0], format="jd").to_datetime()

        return f"{LEVEL1_DIR}/{t.year}/{t.month}/{t.day}/{get_dpc_hashkey(*row[1:])}/{fname}"

    def process(self, filename):
        logger.info(f"Processing {filename} to Level 1.")
        if "VisSci" in filename:
            with VISDALevel0HDUList(filename) as hdulist:
                path = self.get_level1_filename(filename)
                hdulist = hdulist.to_level1()
                # hdulist.writeto(path, overwrite=True, checksum=True)
                logger.info(f"Wrote {filename} to {path}")
        if "VisImg" in filename:
            with VISDAFFILevel0HDUList(filename) as hdulist:
                path = self.get_level1_filename(filename)
                hdulist = hdulist.to_level1()
                # hdulist.writeto(path, overwrite=True, checksum=True)
                logger.info(f"Wrote {filename} to {path}")
        if "InfImg" in filename:
            with NIRDALevel0HDUList(filename) as hdulist:
                path = self.get_level1_filename(filename)
                hdulist = hdulist.to_level1()
                # hdulist.writeto(path, overwrite=True, checksum=True)
                logger.info(f"Wrote {filename} to {path}")

        return self.get_entry(filename)

    def crawl_and_process_parallel(self, max_workers=16, crsoftver=CRSOFTVER):
        paths = self.files_to_process(crsoftver=crsoftver)
        rows = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(self.process, p) for p in paths]
            for fut in as_completed(futures):
                row = fut.result()
                if row is not None:
                    rows.append(row)
        self.add_entries(rows)

    def crawl_and_process(self, crsoftver=None):
        for path in self.files_to_process(crsoftver=CRSOFTVER):
            self.add_entry(self.process(path))


def _process_time(time):
    if isinstance(time, Time):
        time = time.jd
    elif not isinstance(time, float):
        try:
            time = Time(time).jd
        except ValueError:
            raise ValueError("`time_range` must be in JD.")
    return time
