# flake8: noqa W291
"""Tools for keeping a database of pandora files"""

import os
import sqlite3
import stat

import numpy as np
from astropy.time import Time

from . import DPC_KEYS
from .. import CRSOFTVER, LEVEL0_DIR, LEVEL1_DIR, __version__, logger
from ..utils import get_dpc_hashkey
from .mixins import DataBaseMixins, ArchiveDataBaseMixins


class Level1DataBase(ArchiveDataBaseMixins, DataBaseMixins):
    """Database for managing Level 1 files."""

    table_name = "pointings"
    db_path = f"{LEVEL1_DIR}/level1.db"
    level = 1
    level_dir = LEVEL1_DIR

    _sql_key_dict = DPC_KEYS

    def __init__(self):
        super().__init__()
        self.cur.execute(f"ATTACH DATABASE '{LEVEL0_DIR}/level0.db' AS level0")

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

    def check_filename_in_database(self, filename, level=0):
        fname = filename.split("/")[-1] if "/" in filename else filename
        self.cur.execute(
            f"SELECT pfsoftver FROM {f'level{self.level - 1}.' if level == 0 else ''}pointings WHERE filename=?",
            (fname,),
        )
        return self.cur.fetchone() is not None

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
            t = Time(row[13], format="jd").to_datetime()
            targ_id, targ_ra, targ_dec = row[21], row[22], row[23]
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

    def _get_filemap(self):
        from ..visda import VISDAFFILevel0HDUList, VISDALevel0HDUList
        from ..nirda import NIRDALevel0HDUList

        filemap = {
            "VisSci": VISDALevel0HDUList,
            "VisImg": VISDAFFILevel0HDUList,
            "InfImg": NIRDALevel0HDUList,
        }

        return filemap

    def process(self, filename, **kwargs):
        logger.info(f"Processing {filename} to Level {self.level}.")
        logger.info("Ensuring file directory present.")
        path = self.get_output_filename(filename)
        os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
        filemap = self._get_filemap()
        for key, HDUList in filemap.items():
            if key in filename:
                with HDUList(filename) as hdulist:
                    hdulist = getattr(hdulist, f"to_level{self.level}")(**kwargs)
                    hdulist.writeto(path, overwrite=True, checksum=True)
        logger.info(f"Wrote {filename.split('/')[-1]} to {path}")
        return self.get_entry(filename)

    def crawl_and_process(self, crsoftver=CRSOFTVER, nchunks=None, chunk=None):
        paths = self.get_files_to_process(
            crsoftver=crsoftver, nchunks=nchunks, chunk=chunk
        )
        pointings = self.get_pointings_to_process(
            crsoftver=crsoftver, nchunks=nchunks, chunk=chunk
        )
        for pointing, path in zip(pointings, paths):
            try:
                self.add_entry(
                    self.process(
                        path,
                        # targ_ra=pointing[0] if pointing[0] is not None else 0,
                        # targ_dec=pointing[1] if pointing[1] is not None else 0,
                        # targ_rll=pointing[2] if pointing[2] is not None else 40,
                    )
                )
            except:
                logger.exception(
                    f"Error while increasing Level {self.level - 1} to Level {self.level} [{path}]. Skipping."
                )
