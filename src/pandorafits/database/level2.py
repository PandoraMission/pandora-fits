# flake8: noqa W291
"""Tools for keeping a database of pandora files"""

import os
import sqlite3
import stat

from astropy.time import Time

from .. import CRSOFTVER, LEVEL0_DIR, LEVEL1_DIR, LEVEL2_DIR, __version__, logger
from ..nirda import NIRDALevel1HDUList
from ..utils import get_dpc_hashkey
from ..visda import VISDAFFILevel1HDUList, VISDALevel1HDUList
from .level1 import Level1DataBase
from .mixins import DataBaseMixins


class Level2DataBase(Level1DataBase, DataBaseMixins):
    """Database for managing Level 2 files."""

    def __init__(self):
        self.level = 2
        self.level_dir = LEVEL2_DIR
        self.db_path = f"{self.level_dir}/level{self.level}.db"
        self.conn = sqlite3.connect(self.db_path, timeout=120)
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
        self.cur.execute(f"ATTACH DATABASE '{LEVEL0_DIR}/pointings.db' AS level0")
        self.cur.execute(f"ATTACH DATABASE '{LEVEL1_DIR}/level1.db' AS level1")

    # def process(self, filename):
    #     logger.info(f"Processing {filename} to Level {self.level}.")
    #     logger.info("Ensuring file directory present.")
    #     path = self.get_output_filename(filename)
    #     os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
    #     if "VisSci" in filename:
    #         with VISDALevel1HDUList(filename) as hdulist:
    #             logger.info(f"Converting {filename.split('/')[-1]} to Level 2 product.")
    #     #         hdulist = getattr(hdulist, f"to_level{self.level}")()
    #     #         hdulist.writeto(path, overwrite=True, checksum=True)
    #     if "VisImg" in filename:
    #         with VISDAFFILevel1HDUList(filename) as hdulist:
    #             logger.info(f"Converting {filename.split('/')[-1]} to Level 2 product.")
    #     #         hdulist = getattr(hdulist, f"to_level{self.level}")()
    #     #         hdulist.writeto(path, overwrite=True, checksum=True)
    #     if "InfImg" in filename:
    #         with NIRDALevel1HDUList(filename) as hdulist:
    #             logger.info(f"Converting {filename.split('/')[-1]} to Level 2 product.")
    #     #         hdulist = getattr(hdulist, f"to_level{self.level}")()
    #     #         hdulist.writeto(path, overwrite=True, checksum=True)
    #     logger.info(f"Wrote {filename.split('/')[-1]} to {path}")
    #     return self.get_entry(filename)
