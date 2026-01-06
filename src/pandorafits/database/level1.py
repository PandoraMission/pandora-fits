# flake8: noqa W291
"""Tools for keeping a database of pandora files"""

import os
import sqlite3
import stat
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy

import numpy as np
from astropy.time import Time

from .. import CRSOFTVER, LEVEL0_DIR, LEVEL1_DIR, __version__, logger
from ..nirda import NIRDALevel0HDUList, NIRDALevel1HDUList
from ..utils import get_dpc_hashkey
from ..visda import (
    VISDAFFILevel0HDUList,
    VISDAFFILevel1HDUList,
    VISDALevel0HDUList,
    VISDALevel1HDUList,
)
from .mixins import DataBaseMixins, FileDataBaseMixins


class Level1DataBase(FileDataBaseMixins, DataBaseMixins):
    """Database for managing Level 1 files."""

    table_name = "pointings"
    level = 1
    level_dir = LEVEL1_DIR

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
        "targ_rll": "FLOAT",
        "naxis1": "INT",
        "naxis2": "INT",
        "naxis3": "INT",
        "naxis4": "INT",
        "badchecksum": "INT",
        "baddatasum": "INT",
        "filesize": "FLOAT",
    }

    def __init__(self):
        self.db_path = f"{self.level_dir}/level{self.level}.db"
        self.conn = sqlite3.connect(self.db_path, timeout=120)
        self.cur = self.conn.cursor()

        key_string = ", ".join(
            [f"{key} {item}" for key, item in self._sql_key_dict.items()]
        )
        self.cur.execute(
            f"""
        CREATE TABLE IF NOT EXISTS pointings ({key_string})
        """
        )

        key_string = ", ".join([f"{key}" for key, item in self._sql_key_dict.items()])
        value_string = ", ".join(["?"] * len(self._sql_key_dict))
        self.update_str = f"""INSERT INTO pointings 
        ({key_string})
        VALUES ({value_string})"""
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
        return f"Pandora Level{self.level}DataBase"

    @property
    def n_files_to_process(self):
        self.cur.execute(
            f"""SELECT COUNT()
                FROM level{self.level - 1}.pointings AS src
                LEFT JOIN pointings AS dst
                        ON src.filename = dst.filename
                WHERE (src.crsoftver = ? AND src.badchecksum = 0 AND src.baddatasum = 0)
                    AND (dst.filename IS NULL OR dst.pfsoftver != ?);""",
            (CRSOFTVER, __version__),
        )
        nrows = self.cur.fetchone()[0]
        return nrows

    def get_x_to_process(self, x, crsoftver=CRSOFTVER, nchunks=None, chunk=None):
        # Compute limit/offset if chunking is requested
        if nchunks is not None and chunk is not None:
            # total files
            N = self.n_files_to_process
            chunk_size = int(np.ceil(N / nchunks))
            offset = chunk * chunk_size
            limit = chunk_size
        else:
            limit = None
            offset = None

        base_sql = f"""
            SELECT {x}
            FROM level{self.level - 1}.pointings AS src
            LEFT JOIN pointings AS dst
                 ON src.filename = dst.filename
            WHERE (src.crsoftver = ? AND src.badchecksum = 0 AND src.baddatasum = 0)
              AND (dst.filename IS NULL OR dst.pfsoftver != ?)
        """

        params = [crsoftver, __version__]

        # Add LIMIT/OFFSET only if chunking
        if limit is not None:
            base_sql += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        self.cur.execute(base_sql, tuple(params))
        rows = self.cur.fetchall()

        if rows:
            return rows
        else:
            return []

    def get_files_to_process(self, crsoftver=CRSOFTVER, nchunks=None, chunk=None):
        files = self.get_x_to_process(
            x="src.lvldir, src.lvlfilename",
            crsoftver=crsoftver,
            nchunks=nchunks,
            chunk=chunk,
        )
        return [f"{f[0]}/{f[1]}" for f in files]

    def n_files_to_process(self, crsoftver=CRSOFTVER):
        return self.get_x_to_process(
            x="COUNT()", crsoftver=crsoftver, nchunks=None, chunk=None
        )[0]

    def get_pointings_to_process(self, crsoftver=CRSOFTVER, nchunks=None, chunk=None):
        return self.get_x_to_process(
            x="src.targ_ra, src.targ_dec, src.targ_rll",
            crsoftver=crsoftver,
            nchunks=nchunks,
            chunk=chunk,
        )

    # def _is_bad_value(
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
    #             if level == self.level:
    #                 self.cur.execute("DELETE FROM pointings WHERE filename=?", (fname,))
    #             else:
    #                 raise ValueError(f"Can not parse usage of level `{level}`")
    #             self.conn.commit()
    #             logger.warning(f"File {fname} removed from Level {level} database.")
    #         return True
    #     return False

    # def _is_bad_row(
    #     self,
    #     filename,
    #     crsoftver=CRSOFTVER,
    #     pfsoftver=__version__,
    #     badchecksum=0,
    #     baddatasum=0,
    #     level=0,
    #     remove_bad_entries=False,
    # ):
    #     fname = filename.split("/")[-1] if "/" in filename else filename
    #     self.cur.execute(
    #         f"SELECT crsoftver, pfsoftver, badchecksum, baddatasum FROM {'level0.' if level == 0 else ''}pointings WHERE filename=?",
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
    #     is_bad_row = False
    #     # Log any discrepancies
    #     if crsoftver is not None:
    #         is_bad_row |= self._is_bad_value(
    #             fname=fname,
    #             expected_entry=crsoftver,
    #             entry=database_crsoftver,
    #             name="CRSOFTV",
    #             level=0,
    #             remove_bad_entries=remove_bad_entries,
    #         )
    #     if pfsoftver is not None:
    #         if database_pfsoftver is not None:
    #             is_bad_row |= self._is_bad_value(
    #                 fname=fname,
    #                 expected_entry=pfsoftver,
    #                 entry=database_pfsoftver,
    #                 name="PFSOFTV",
    #                 level=0,
    #                 remove_bad_entries=remove_bad_entries,
    #             )
    #     if badchecksum is not None:
    #         is_bad_row |= self._is_bad_value(
    #             fname=fname,
    #             expected_entry=badchecksum,
    #             entry=database_badchecksum,
    #             name="badchecksum",
    #             level=0,
    #             remove_bad_entries=remove_bad_entries,
    #         )
    #     if baddatasum is not None:
    #         is_bad_row |= self._is_bad_value(
    #             fname=fname,
    #             expected_entry=baddatasum,
    #             entry=database_baddatasum,
    #             name="baddatasum",
    #             level=0,
    #             remove_bad_entries=remove_bad_entries,
    #         )
    #     return is_bad_row

    def check_filename_in_database(self, filename, level=0):
        fname = filename.split("/")[-1] if "/" in filename else filename
        self.cur.execute(
            f"SELECT pfsoftver FROM {f'level{self.level - 1}.' if level == 0 else ''}pointings WHERE filename=?",
            (fname,),
        )
        return self.cur.fetchone() is not None

    # def files_to_process(self, crsoftver=CRSOFTVER, nchunks=1, chunknumber=1):
    #     cur = self.conn.cursor()
    #     cur.execute("SELECT filename, dir FROM level0.pointings")

    #     row = cur.fetchone()
    #     while row is not None:
    #         if (not self.check_filename_in_database(row[0], level=self.level)) & (
    #             not self._is_bad_row(row[0], crsoftver=crsoftver, level=0)
    #         ):
    #             yield (f"{row[1]}/{row[0]}")
    #         row = cur.fetchone()

    def get_entry(self, filename):
        fname = filename.split("/")[-1] if "/" in filename else filename
        self.cur.execute(
            f"SELECT * FROM level{self.level - 1}.pointings WHERE lvlfilename=?",
            (fname,),
        )
        row = self.cur.fetchone()
        output_filename = self.get_output_filename(row)
        lvldir, lvlfilename = (
            "/".join(output_filename.split("/")[:-1]),
            output_filename.split("/")[-1],
        )
        return (row[0], lvlfilename, row[2], lvldir, *row[4:])

    def get_output_filename(self, filename_or_row):
        if isinstance(filename_or_row, tuple):
            row = filename_or_row
            t = Time(row[12], format="jd").to_datetime()
            targ_id, targ_ra, targ_dec = row[20], row[21], row[22]
            fname = row[0]
        elif isinstance(filename_or_row, str):
            # filename = filename_or_row
            fname = (
                filename_or_row.split("/")[-1]
                if "/" in filename_or_row
                else filename_or_row
            )
            self.cur.execute(
                f"SELECT filename, start, targ_id, targ_ra, targ_dec FROM level{self.level - 1}.pointings WHERE lvlfilename=?",
                (fname,),
            )
            row = self.cur.fetchone()
            t = Time(row[1], format="jd").to_datetime()
            targ_id, targ_ra, targ_dec = row[2:]
            fname = row[0]
            if row is None:
                return None
        elif filename_or_row is None:
            return None
        fname_no_suffix = ".".join(fname.split(".")[:-1])
        suffix = fname.split(".")[-1]
        return f"{self.level_dir}/{t.year}/{t.month}/{t.day}/{get_dpc_hashkey(targ_id, targ_ra, targ_dec)}/{fname_no_suffix}_v{__version__.replace('.', '-')}_l{self.level}.{suffix}"

    def process(self, filename, **kwargs):
        logger.info(f"Processing {filename} to Level {self.level}.")
        logger.info("Ensuring file directory present.")
        path = self.get_output_filename(filename)
        os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
        filemap = {
            "VisSci": globals()[f"VISDALevel{self.level - 1}HDUList"],
            "VisImg": globals()[f"VISDAFFILevel{self.level - 1}HDUList"],
            "InfImg": globals()[f"NIRDALevel{self.level - 1}HDUList"],
        }
        for key, HDUList in filemap.items():
            if key in filename:
                try:
                    with HDUList(filename) as hdulist:
                        hdulist = getattr(hdulist, f"to_level{self.level}")(**kwargs)
                        hdulist.writeto(path, overwrite=True, checksum=True)
                except:
                    logger.exception(
                        f"Error while creating {HDUList.__name__} file. Skipping."
                    )
                    return None

        logger.info(f"Wrote {filename.split('/')[-1]} to {path}")
        return self.get_entry(filename)

    # def crawl_and_process_parallel(self, max_workers=16, crsoftver=CRSOFTVER):
    #     paths = list(self.files_to_process(crsoftver=crsoftver))
    #     for path in paths:
    #         out_path = self.get_output_filename(path)
    #         os.makedirs("/".join(out_path.split("/")[:-1]), exist_ok=True)
    #     rows = []
    #     with ThreadPoolExecutor(max_workers=max_workers) as ex:
    #         futures = [ex.submit(self.process, p) for p in paths]
    #         for fut in as_completed(futures):
    #             row = fut.result()
    #             if row is not None:
    #                 rows.append(row)
    #     self.add_entries(rows)

    def crawl_and_process(self, crsoftver=CRSOFTVER, nchunks=None, chunk=None):
        paths = self.get_files_to_process(
            crsoftver=crsoftver, nchunks=nchunks, chunk=chunk
        )
        pointings = self.get_pointings_to_process(
            crsoftver=crsoftver, nchunks=nchunks, chunk=chunk
        )
        for pointing, path in zip(pointings, paths):
            self.add_entry(
                self.process(
                    path,
                    targ_ra=pointing[0] if pointing[0] is not None else 0,
                    targ_dec=pointing[1] if pointing[1] is not None else 0,
                    targ_rll=pointing[2] if pointing[2] is not None else 0,
                )
            )

    # def crawl_and_process_parallel(self, max_workers=16, crsoftver=CRSOFTVER):
    #     paths = list(self.files_to_process(crsoftver=crsoftver))
    #     for path in paths:
    #         out_path = self.get_output_filename(path)
    #         os.makedirs("/".join(out_path.split("/")[:-1]), exist_ok=True)
    #     rows = []
    #     with ThreadPoolExecutor(max_workers=max_workers) as ex:
    #         futures = [ex.submit(self.process, p) for p in paths]
    #         for fut in as_completed(futures):
    #             row = fut.result()
    #             if row is not None:
    #                 rows.append(row)
    #     self.add_entries(rows)
