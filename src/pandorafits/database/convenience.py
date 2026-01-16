import logging
import os
import shutil

import numpy as np
import pandas as pd

from .. import (
    CRSOFTVER,
    LEVEL0_DIR,
    LEVEL1_DIR,
    LEVEL2_DIR,
    LEVEL3_DIR,
    LOG_DIR,
    __version__,
    logger,
)
from . import Level1DataBase  # noqa
from . import Level2DataBase  # noqa
from . import Level3DataBase  # noqa
from . import AstrometryDataBase, Level0DataBase, TargetDataBase

__all__ = [
    "delete_targetdatabase",
    "update_targetdatabase",
    "delete_astrometrydatabase",
    "update_astrometrydatabase",
    "update_level0database",
    "delete_level0database",
    "delete_level1database",
    "delete_level2database",
    "delete_level3database",
    "delete_level1filestorage",
    "delete_level2filestorage",
    "delete_level3filestorage",
    "delete_logs",
    "delete_all",
    "get_status",
    "get_level_database",
    "get_astrometry_database",
    "get_target_database",
    "get_level_paths",
]


def delete_targetdatabase() -> None:
    """
    Deletes the SQLite database file for targets.

    Raises
    ------
    FileNotFoundError
        If the database file does not exist.
    """
    db_path = f"{LEVEL0_DIR}/targets.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.warning(f"Target Database at {db_path} has been deleted.")
    else:
        logger.warning(
            f"Tried to delete TargetDataBase. No database found at {db_path}."
        )


def update_targetdatabase() -> None:
    """
    Creates and updates to the SQLite database file.
    """
    with TargetDataBase() as db:
        db.crawl_and_add()


def delete_astrometrydatabase() -> None:
    """
    Deletes the SQLite database file for astrometry.

    Raises
    ------
    FileNotFoundError
        If the database file does not exist.
    """
    db_path = f"{LEVEL0_DIR}/astrometry.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.warning(f"Astrometry Database at {db_path} has been deleted.")
    else:
        logger.warning(
            f"Tried to delete AstrometryDataBase. No database found at {db_path}."
        )


def update_astrometrydatabase() -> None:
    """
    Creates and updates to the SQLite database file.
    """
    with AstrometryDataBase() as db:
        db.crawl_and_add()


def update_level0database() -> None:
    """
    Creates and updates to the SQLite database file.
    """
    with Level0DataBase() as db:
        db.crawl_and_add()


def delete_level0database() -> None:
    """
    Deletes the SQLite database file.

    Raises
    ------
    FileNotFoundError
        If the database file does not exist.
    """
    db_path = f"{LEVEL0_DIR}/level0.db"
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


def delete_logs():
    logger.info("Running `delete_logs`")
    current_logfile = next(
        h.baseFilename for h in logger.handlers if isinstance(h, logging.FileHandler)
    )
    # Ensure directory exists
    os.makedirs(LOG_DIR, exist_ok=True)

    # Iterate over all files in the log directory
    for fname in os.listdir(LOG_DIR):
        fpath = os.path.join(LOG_DIR, fname)

        # Skip subdirs and skip the current logfile
        if not os.path.isfile(fpath):
            continue
        if os.path.abspath(fpath) == os.path.abspath(current_logfile):
            continue

        # Delete everything else
        os.remove(fpath)
    logger.info("Finished `delete_logs`")


def delete_all():
    """Clear all processing data"""
    delete_logs()
    delete_astrometrydatabase()
    delete_level0database()
    delete_level1database()
    delete_level1filestorage()
    delete_level2database()
    delete_level2filestorage()
    delete_level3database()
    delete_level3filestorage()


def get_status():
    def get_nrows(name):
        with globals()[name]() as self:
            self.cur.execute(f"""SELECT COUNT() FROM {self.table_name}""")
            nrows = self.cur.fetchone()[0]
        return nrows

    def get_ntargets(name):
        with globals()[name]() as self:
            self.cur.execute(
                f"""SELECT COUNT(DISTINCT targ_id) FROM {self.table_name}"""
            )
            nrows = self.cur.fetchone()[0]
        return nrows

    def get_npointings(name):
        with globals()[name]() as self:
            self.cur.execute(f"""SELECT COUNT(DISTINCT start) FROM {self.table_name}""")
            nrows = self.cur.fetchone()[0]
        return nrows

    def get_nbadcrsoftver(name):
        with globals()[name]() as self:
            self.cur.execute(
                f"""SELECT COUNT() FROM {self.table_name} WHERE crsoftver != ?""",
                (CRSOFTVER,),
            )
            nrows = self.cur.fetchone()[0]
        return nrows

    def get_nbadpfsoftver(name):
        with globals()[name]() as self:
            self.cur.execute(
                f"""SELECT COUNT() FROM {self.table_name} WHERE pfsoftver != ?""",
                (__version__,),
            )
            nrows = self.cur.fetchone()[0]
        return nrows

    def get_nbadchecksum(name):
        with globals()[name]() as self:
            self.cur.execute(
                f"""SELECT COUNT() FROM {self.table_name} WHERE badchecksum != ?""",
                (0,),
            )
            nrows = self.cur.fetchone()[0]
        return nrows

    def get_nbaddatasum(name):
        with globals()[name]() as self:
            self.cur.execute(
                f"""SELECT COUNT() FROM {self.table_name} WHERE baddatasum != ?""", (0,)
            )
            nrows = self.cur.fetchone()[0]
        return nrows

    names = ["AstrometryDataBase", *[f"Level{level}DataBase" for level in np.arange(4)]]
    r = {
        "n_row_s": [get_nrows(name) for name in names],
        "n_bad_crsoftver": [get_nbadcrsoftver(name) for name in names],
        "n_bad_pfsoftver": [get_nbadpfsoftver(name) for name in names],
        "n_bad_checksum": [get_nbadchecksum(name) for name in names],
        "n_bad_datasum": [get_nbaddatasum(name) for name in names],
        "n_unique_targets": [get_ntargets(name) for name in names],
        # "n_unique_pointings": [get_npointings(name) for name in names],
    }
    status = pd.DataFrame.from_dict(r).T
    status.columns = names
    return status


def get_level_database(level):
    with globals()[f"Level{level}DataBase"]() as self:
        df = self.to_pandas()
    return df


def get_astrometry_database():
    with AstrometryDataBase() as self:
        df = self.to_pandas()
    return df


def get_target_database():
    with TargetDataBase() as self:
        df = self.to_pandas()
    return df


def get_level_paths(level=0):
    df = get_level_database(level)
    return (df.lvldir + "/" + df.lvlfilename).values
