# flake8: noqa W291
"""Tools for keeping a database of pandora files"""

# Standard library
import os
import sqlite3
import stat
from pathlib import Path

# Third-party
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

from .. import LEVEL0_DIR, LEVELE_DIR, __version__, logger
from ..utils import get_dpc_hashkey
from . import BANSTRINGS
from .engineering import EngineeringDataBase, PayloadDataBase
from .mixins import DataBaseMixins


def get_engineering_fits(time_range):
    pri = fits.PrimaryHDU()
    pri.header["PFTIME"] = (
        Time.now().isot,
        "Pandora DPC Processing Time",
    )
    hdulist = [pri]
    with EngineeringDataBase() as self:
        df = self.to_pandas(time_range=time_range)
    hdulist.append(fits.BinTableHDU(Table.from_pandas(df), name="ENGINEERING"))
    with PayloadDataBase() as self:
        df = self.to_pandas(time_range=time_range)
    hdulist.append(fits.BinTableHDU(Table.from_pandas(df), name="PAYLOAD"))
    return fits.HDUList(hdulist)


class LevelEDataBase(DataBaseMixins):
    """Database for managing Level 1 files."""

    table_name = "pointings"
    db_path = os.path.join(LEVELE_DIR, "levele.db")
    level = "E"
    level_dir = LEVELE_DIR

    _sql_key_dict = {
        "filename": "TEXT PRIMARY KEY",
        "lvlfilename": "TEXT",
        "dir": "TEXT",
        "lvldir": "TEXT",
        "jd": "FLOAT",
        "create_time": "FLOAT",
    }

    def __init__(self):
        super().__init__()
        self.cur.execute(
            f"ATTACH DATABASE '{os.path.join(LEVEL0_DIR, 'level0.db')}' AS level0"
        )

    def __repr__(self):
        return f"Pandora Level{self.level}DataBase"

    @property
    def n_files_to_process(self):
        self.cur.execute(
            f"""SELECT COUNT()
                FROM level0.pointings AS src
                LEFT JOIN pointings AS dst
                        ON src.filename = dst.filename
                WHERE (src.badchecksum = 0 AND src.baddatasum = 0)
                    AND (dst.filename IS NULL);""",
        )
        nrows = self.cur.fetchone()[0]
        return nrows

    def get_x_to_process(self, x, nchunks=None, chunk=None):
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
            FROM level0.pointings AS src
            LEFT JOIN pointings AS dst
                 ON src.filename = dst.filename
            WHERE (src.badchecksum = 0 AND src.baddatasum = 0)
              AND (dst.filename IS NULL)
              OR ((dst.create_time - dst.jd) < 15)
        """

        params = []  # [__version__]

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

    def get_files_to_process(self, nchunks=None, chunk=None):
        files = self.get_x_to_process(
            x="src.lvldir, src.lvlfilename",
            nchunks=nchunks,
            chunk=chunk,
        )
        return [os.path.join(f[0], f[1]) for f in files]

    # def n_files_to_process(self):
    #     return self.get_x_to_process(x="COUNT()", nchunks=None, chunk=None)[0]

    # def get_pointings_to_process(self, nchunks=None, chunk=None):
    #     return self.get_x_to_process(
    #         x="src.targ_ra, src.targ_dec, src.targ_rll",
    #         nchunks=nchunks,
    #         chunk=chunk,
    #     )

    def check_filename_needs_processing(self, filename, level=0):
        fname = os.path.basename(filename)
        fname = Path(filename).name
        if np.any([ban in fname for ban in BANSTRINGS]):
            # Automatically skip
            return False

        self.cur.execute(
            f"SELECT pfsoftver FROM 'level0.pointings WHERE filename=?",
            (fname,),
        )
        return self.cur.fetchone() is None

    def get_entry(self, filename):
        fname = os.path.basename(filename)
        self.cur.execute(
            f"SELECT * FROM level0.pointings WHERE lvlfilename=?",
            (fname,),
        )
        row = self.cur.fetchone()
        output_filename = self.get_output_filename(filename)
        lvldir, lvlfilename = os.path.split(output_filename)
        return (row[0], lvlfilename, row[2], lvldir, row[8], Time.now().jd)

    def get_output_filename(self, filename):
        fname = os.path.basename(filename)
        self.cur.execute(
            f"SELECT filename, start, targ_id, targ_ra, targ_dec FROM level0.pointings WHERE lvlfilename=?",
            (fname,),
        )
        row = self.cur.fetchone()
        if row is None:
            return None
        t = Time(row[1], format="jd").to_datetime()
        targ_id, targ_ra, targ_dec = row[2:]
        fname = row[0]

        fname_no_suffix, suffix = os.path.splitext(fname)
        return os.path.join(
            self.level_dir,
            str(t.year),
            str(t.month),
            str(t.day),
            get_dpc_hashkey(targ_id, targ_ra, targ_dec),
            f"{fname_no_suffix}_v{__version__.replace('.', '-')}_eng{suffix}",
        )

    def process(self, filename, **kwargs):
        logger.info(f"Processing {filename} engineering data.")
        logger.info("Ensuring file directory present.")
        path = self.get_output_filename(filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fname = os.path.basename(filename)
        self.cur.execute(
            f"SELECT start, exptime FROM level0.pointings WHERE lvlfilename=?",
            (fname,),
        )
        start, exptime = self.cur.fetchone()
        hdulist = get_engineering_fits(
            (start - 600 / 86400, start + exptime / 86400 + 600 / 86400)
        )

        hdulist.writeto(path, overwrite=True, checksum=True)
        logger.info(f"Wrote {os.path.basename(filename)} to {path}")
        return self.get_entry(filename)

    def crawl_and_process(self, nchunks=None, chunk=None):
        paths = self.get_files_to_process(nchunks=nchunks, chunk=chunk)
        for path in paths:
            # try:
            entry = self.process(
                path,
            )
            self.add_entry(entry)

            # except:
            #     logger.exception(f"Error while creating engineering file. Skipping.")
