# flake8: noqa W291
"""Tools for keeping a database of pandora files"""

from .. import LEVEL1_DIR, LEVEL2_DIR, __version__
from .level1 import Level1DataBase
from .mixins import DataBaseMixins, FileDataBaseMixins


class Level2DataBase(Level1DataBase, FileDataBaseMixins, DataBaseMixins):
    """Database for managing Level 2 files."""

    table_name = "pointings"
    level = 2
    level_dir = LEVEL2_DIR

    def __init__(self):
        super().__init__()
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
