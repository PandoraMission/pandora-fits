# flake8: noqa W291
"""Tools for keeping a database of pandora files"""

from .. import LEVEL3_DIR, LEVEL2_DIR, LEVEL1_DIR, __version__, LEVEL0_DIR
from .mixins import DataBaseMixins
from .level1 import Level1DataBase

import os
import sqlite3
import stat

from astropy.time import Time

from .. import LEVEL0_DIR, logger, LEVEL1_DIR, __version__, CRSOFTVER
from ..visda import VISDAFFILevel0HDUList, VISDALevel0HDUList
from ..nirda import NIRDALevel0HDUList
from ..utils import get_dpc_hashkey
from .mixins import DataBaseMixins


class Level3DataBase(Level1DataBase, DataBaseMixins):
    """Database for managing Level 2 files."""

    def __init__(self):
        self.level = 3
        self.level_dir = LEVEL3_DIR
        self.db_path = f"{self.level_dir}/level{self.level}.db"
        self.conn = sqlite3.connect(self.db_path, timeout=120)
        self.cur = self.conn.cursor()
        self.cur.execute(
            """
        CREATE TABLE IF NOT EXISTS pointings (
            filename TEXT PRIMARY KEY,
            lvlfilename TEXT,
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
        (filename, lvlfilename, dir, crsoftver, pfsoftver, finetime, corstime,
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
        self.cur.execute(f"ATTACH DATABASE '{LEVEL2_DIR}/level2.db' AS level2")

    def get_output_filename(self, filename_or_row):
        if isinstance(filename_or_row, tuple):
            fname = filename_or_row[0]
        elif isinstance(filename_or_row, str):
            filename = filename_or_row
            fname = filename.split("/")[-1] if "/" in filename else filename
        elif filename_or_row is None:
            return None
        self.cur.execute(
            f"SELECT targ_id, jd, ra, dec FROM level{self.level - 1}.pointings WHERE jd = (SELECT start FROM level{self.level - 1}.pointings WHERE filename=?)",
            (fname,),
        )
        row = self.cur.fetchone()
        if row is None:
            return None
        t = Time(row[1], format="jd").to_datetime()
        return f"{self.level_dir}/{t.year}/{t.month}/{t.day}/{get_dpc_hashkey(row[0], row[2], row[3])}/{Time(row[1], format='jd').strftime('%Y-%m-%d__%H-%M-%S')}_{row[0]}_v{__version__.replace('.', '-')}_l3.fits"

    def process(self, filename):
        logger.info(f"Processing {filename} to Level {self.level}.")
        logger.info("Ensuring file directory present.")
        path = self.get_output_filename(filename)
        os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
        # Somehow write data to the level 3 file here...
        # if "VisSci" in filename:
        #     with VISDALevel0HDUList(filename) as hdulist:
        #         hdulist = getattr(hdulist, f"to_level{self.level}")()
        #         hdulist.writeto(path, overwrite=True, checksum=True)
        # if "InfImg" in filename:
        #     with NIRDALevel0HDUList(filename) as hdulist:
        #         hdulist = getattr(hdulist, f"to_level{self.level}")()
        #         hdulist.writeto(path, overwrite=True, checksum=True)
        if "VisImg" in filename:
            # VisImg are full frame images, they don't go into our level 3 products.
            logger.warning("Can not process VisImg to Level 3. Will skip this file.")
            return None
        logger.info(f"Wrote {filename.split('/')[-1]} to {path}")
        return self.get_entry(filename)
