# flake8: noqa W291
"""Tools for keeping a database of pandora files"""

import os
import sqlite3
import stat

from astropy.time import Time

from .. import (
    CRSOFTVER,
    LEVEL0_DIR,
    LEVEL1_DIR,
    LEVEL2_DIR,
    LEVEL3_DIR,
    __version__,
    logger,
)
from ..nirda import NIRDALevel0HDUList
from ..utils import get_dpc_hashkey
from ..visda import VISDAFFILevel0HDUList, VISDALevel0HDUList
from .level1 import Level1DataBase
from .mixins import DataBaseMixins, FileDataBaseMixins


class Level3DataBase(Level1DataBase, FileDataBaseMixins, DataBaseMixins):
    """Database for managing Level 2 files."""

    table_name = "pointings"
    level = 3
    level_dir = LEVEL3_DIR

    def __init__(self):
        super().__init__()
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

    # def process(self, filename):
    #     logger.info(f"Processing {filename} to Level {self.level}.")
    #     logger.info("Ensuring file directory present.")
    #     path = self.get_output_filename(filename)
    #     os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
    #     # Somehow write data to the level 3 file here...
    #     # if "VisSci" in filename:
    #     #     with VISDALevel0HDUList(filename) as hdulist:
    #     #         hdulist = getattr(hdulist, f"to_level{self.level}")()
    #     #         hdulist.writeto(path, overwrite=True, checksum=True)
    #     # if "InfImg" in filename:
    #     #     with NIRDALevel0HDUList(filename) as hdulist:
    #     #         hdulist = getattr(hdulist, f"to_level{self.level}")()
    #     #         hdulist.writeto(path, overwrite=True, checksum=True)
    #     if "VisImg" in filename:
    #         # VisImg are full frame images, they don't go into our level 3 products.
    #         logger.warning("Can not process VisImg to Level 3. Will skip this file.")
    #         return None
    #     logger.info(f"Wrote {filename.split('/')[-1]} to {path}")
    #     return self.get_entry(filename)
