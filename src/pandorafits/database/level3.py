# flake8: noqa W291
"""Tools for keeping a database of pandora files"""

from astropy.time import Time

from .. import LEVEL2_DIR, LEVEL3_DIR, __version__
from ..utils import get_dpc_hashkey
from .level2 import Level2DataBase


class Level3DataBase(Level2DataBase):
    """Database for managing Level 2 files."""

    table_name = "pointings"
    db_path = f"{LEVEL3_DIR}/level3.db"
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

    def _get_filemap(self):
        from ..nirda import NIRDALevel2HDUList
        from ..visda import VISDAFFILevel2HDUList, VISDALevel2HDUList

        filemap = {
            "VisSci": VISDALevel2HDUList,
            "VisImg": VISDAFFILevel2HDUList,
            "InfImg": NIRDALevel2HDUList,
        }

        return filemap
