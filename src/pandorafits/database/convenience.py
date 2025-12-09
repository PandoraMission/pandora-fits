import os
import shutil

import numpy as np
import pandas as pd

from .. import (
    CRSOFTVER,
    DATA_DIR,
    LEVEL0_DIR,
    LEVEL1_DIR,
    LEVEL2_DIR,
    LEVEL3_DIR,
    __version__,
    logger,
)
from . import Level0DataBase, Level1DataBase, Level2DataBase, Level3DataBase  # noqa

__all__ = [
    "update_filedatabase",
    "delete_filedatabase",
    "delete_level1database",
    "delete_level2database",
    "delete_level1filestorage",
    "delete_level2filestorage",
    "delete_all",
    "get_status",
]


def update_filedatabase() -> None:
    """
    Creates and updates to the SQLite database file.
    """
    with Level0DataBase() as db:
        db.crawl_and_add_parallel(DATA_DIR)
        db.update_pointings()


def delete_filedatabase() -> None:
    """
    Deletes the SQLite database file.

    Raises
    ------
    FileNotFoundError
        If the database file does not exist.
    """
    db_path = f"{LEVEL0_DIR}/pointings.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.warning(f"Database at {db_path} has been deleted.")
    else:
        logger.warning(
            f"Tried to delete Level0DataBase. No database found at {db_path}."
        )


def delete_level1database() -> None:
    """
    Deletes the SQLite database file.

    Raises
    ------
    FileNotFoundError
        If the database file does not exist.
    """
    db_path = f"{LEVEL1_DIR}/level1.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.warning(f"Database at {db_path} has been deleted.")
    else:
        logger.warning(
            f"Tried to delete Level1DataBase. No database found at {db_path}."
        )


def delete_level1filestorage():
    logger.info("Running `delete_level1filestorage`")
    if os.path.exists(LEVEL1_DIR):
        shutil.rmtree(LEVEL1_DIR)
    if not os.path.exists(LEVEL1_DIR):
        os.makedirs(LEVEL1_DIR, exist_ok=True)
        os.chmod(LEVEL1_DIR, 0o750)
    logger.info("Finished `delete_level1filestorage`")


def delete_level2database() -> None:
    """
    Deletes the SQLite database file.

    Raises
    ------
    FileNotFoundError
        If the database file does not exist.
    """
    db_path = f"{LEVEL2_DIR}/level2.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.warning(f"Database at {db_path} has been deleted.")
    else:
        logger.warning(
            f"Tried to delete Level2DataBase. No database found at {db_path}."
        )


def delete_level2filestorage():
    logger.info("Running `delete_level2filestorage`")
    if os.path.exists(LEVEL2_DIR):
        shutil.rmtree(LEVEL2_DIR)
    if not os.path.exists(LEVEL2_DIR):
        os.makedirs(LEVEL2_DIR, exist_ok=True)
        os.chmod(LEVEL2_DIR, 0o750)
    logger.info("Finished `delete_level2filestorage`")


def delete_level3database() -> None:
    """
    Deletes the SQLite database file.

    Raises
    ------
    FileNotFoundError
        If the database file does not exist.
    """
    db_path = f"{LEVEL3_DIR}/level3.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.warning(f"Database at {db_path} has been deleted.")
    else:
        logger.warning(
            f"Tried to delete Level3DataBase. No database found at {db_path}."
        )


def delete_level3filestorage():
    logger.info("Running `delete_level3filestorage`")
    if os.path.exists(LEVEL3_DIR):
        shutil.rmtree(LEVEL3_DIR)
    if not os.path.exists(LEVEL3_DIR):
        os.makedirs(LEVEL3_DIR, exist_ok=True)
        os.chmod(LEVEL3_DIR, 0o750)
    logger.info("Finished `delete_level3filestorage`")


def delete_all():
    """Clear all processing data"""
    delete_filedatabase()
    delete_level1database()
    delete_level1filestorage()
    delete_level2filestorage()
    delete_level2filestorage()
    delete_level3filestorage()
    delete_level3filestorage()


def get_status():
    def get_nrows(level):
        with globals()[f"Level{level}DataBase"]() as self:
            self.cur.execute("""SELECT COUNT() FROM pointings""")
            nrows = self.cur.fetchone()[0]
        return nrows

    def get_ntargets(level):
        with globals()[f"Level{level}DataBase"]() as self:
            self.cur.execute("""SELECT COUNT(DISTINCT targ_id) FROM pointings""")
            nrows = self.cur.fetchone()[0]
        return nrows

    def get_npointings(level):
        with globals()[f"Level{level}DataBase"]() as self:
            self.cur.execute("""SELECT COUNT(DISTINCT start) FROM pointings""")
            nrows = self.cur.fetchone()[0]
        return nrows

    def get_nbadcrsoftver(level):
        with globals()[f"Level{level}DataBase"]() as self:
            self.cur.execute(
                """SELECT COUNT() FROM pointings WHERE crsoftver != ?""",
                (CRSOFTVER,),
            )
            nrows = self.cur.fetchone()[0]
        return nrows

    def get_nbadpfsoftver(level):
        with globals()[f"Level{level}DataBase"]() as self:
            self.cur.execute(
                """SELECT COUNT() FROM pointings WHERE pfsoftver != ?""",
                (__version__,),
            )
            nrows = self.cur.fetchone()[0]
        return nrows

    def get_nbadchecksum(level):
        with globals()[f"Level{level}DataBase"]() as self:
            self.cur.execute(
                """SELECT COUNT() FROM pointings WHERE badchecksum != ?""", (0,)
            )
            nrows = self.cur.fetchone()[0]
        return nrows

    def get_nbaddatasum(level):
        with globals()[f"Level{level}DataBase"]() as self:
            self.cur.execute(
                """SELECT COUNT() FROM pointings WHERE baddatasum != ?""", (0,)
            )
            nrows = self.cur.fetchone()[0]
        return nrows

    r = {
        "n_row_s": [get_nrows(level) for level in np.arange(4)],
        "n_bad_crsoftver": [get_nbadcrsoftver(level) for level in np.arange(4)],
        "n_bad_pfsoftver": [get_nbadpfsoftver(level) for level in np.arange(4)],
        "n_bad_checksum": [get_nbadchecksum(level) for level in np.arange(4)],
        "n_bad_datasum": [get_nbaddatasum(level) for level in np.arange(4)],
        "n_unique_targets": [get_ntargets(level) for level in np.arange(4)],
        "n_unique_pointings": [get_npointings(level) for level in np.arange(4)],
    }
    return pd.DataFrame.from_dict(r).T
