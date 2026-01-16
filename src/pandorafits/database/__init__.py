"""Submodule for working with file database."""

DPC_KEYS = {
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
    "dpc_seq_id": "INT",
    "dpc_hash_key": "TEXT",
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
    "targ_rll_type": "TEXT",
    "ra": "FLOAT",
    "dec": "FLOAT",
    "roll": "FLOAT",
    "naxis1": "INT",
    "naxis2": "INT",
    "naxis3": "INT",
    "naxis4": "INT",
    "badchecksum": "INT",
    "baddatasum": "INT",
    "filesize": "FLOAT",
}


from ..roll import get_roll  # noqa
from .targets import TargetDataBase  # noqa
from .astrometry import AstrometryDataBase  # noqa
from .level0 import Level0DataBase  # noqa
from .level1 import Level1DataBase  # noqa
from .level2 import Level2DataBase  # noqa
from .level3 import Level3DataBase  # noqa
from .convenience import *  # noqa
