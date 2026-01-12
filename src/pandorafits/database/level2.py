# flake8: noqa W291
"""Tools for keeping a database of pandora files"""

from .. import LEVEL1_DIR, LEVEL2_DIR, __version__
from .level1 import Level1DataBase


class Level2DataBase(Level1DataBase):
    """Database for managing Level 2 files."""

    table_name = "pointings"
    db_path = f"{LEVEL2_DIR}/level2.db"
    level = 2
    level_dir = LEVEL2_DIR

    def __init__(self):
        super().__init__()
        self.cur.execute(f"ATTACH DATABASE '{LEVEL1_DIR}/level1.db' AS level1")

    def _get_filemap(self):
        from ..nirda import NIRDALevel1HDUList
        from ..visda import VISDAFFILevel1HDUList, VISDALevel1HDUList

        filemap = {
            "VisSci": VISDALevel1HDUList,
            "VisImg": VISDAFFILevel1HDUList,
            "InfImg": NIRDALevel1HDUList,
        }

        return filemap
