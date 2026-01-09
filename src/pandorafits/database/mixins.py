"""Mixins for databases"""

import os
from datetime import datetime, timezone
from .. import LEVEL0_DIR, LEVEL1_DIR, LEVEL2_DIR, LEVEL3_DIR  # noqa
import stat
import numpy as np
import pandas as pd
from astropy.time import Time
import sqlite3


from .. import logger  # noqa


def _process_time(time):
    if isinstance(time, Time):
        time = time.jd
    elif not isinstance(time, float):
        try:
            time = Time(time).jd
        except ValueError:
            raise ValueError("`time_range` must be in JD.")
    return time


class DataBaseMixins:
    def __init__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cur = self.conn.cursor()

        key_string = ", ".join(
            [f"{key} {item}" for key, item in self._sql_key_dict.items()]
        )
        self.cur.execute(
            f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} ({key_string})
        """
        )

        key_string = ", ".join([f"{key}" for key, item in self._sql_key_dict.items()])
        value_string = ", ".join(["?"] * len(self._sql_key_dict))
        self.update_str = f"""INSERT INTO {self.table_name} 
        ({key_string})
        VALUES ({value_string})"""
        self.conn.commit()

        os.chmod(
            self.db_path,
            stat.S_IRUSR
            | stat.S_IWUSR  # owner: read/write
            | stat.S_IRGRP
            | stat.S_IWGRP,  # group: read/write
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()

    def add_entry(self, values):
        if values is not None:
            self.cur.execute(
                self.update_str,
                values,
            )
            self.conn.commit()

    def add_entries(self, values):
        self.cur.executemany(
            self.update_str,
            [v for v in values if v is not None],
        )
        self.conn.commit()

    def to_pandas(self, time_range=None, **kwargs):
        sql = f"SELECT * FROM {self.table_name}"
        params = []

        where_clauses = []

        if time_range is not None:
            start, end = _process_time(time_range[0]), _process_time(time_range[1])
            start, end = np.sort([start, end])
            where_clauses.append("jd BETWEEN ? AND ?")
            params.extend([start, end])

        for kwarg in kwargs.items():
            if kwarg[1] is not None:
                where_clauses.append(f"{kwarg[0]} = ?")
                params.append(kwarg[1])

        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        return pd.read_sql_query(sql, self.conn, params=params)

    def check_filename_in_database(self, filename):
        self.cur.execute(
            f"SELECT 1 FROM {self.table_name} WHERE filename=?",
            ((filename.split("/")[-1] if "/" in filename else filename),),
        )
        return self.cur.fetchone() is not None


class ArchiveDataBaseMixins:
    def to_archive_manifest(self):
        logger.info(f"Creating Level {self.level} archive manifest.")
        manifest_path = (
            locals()[f"LEVEL{self.level}_DIR"] + "/" + f"level{self.level}_manifest.csv"
        )
        columns = [
            "Target Name",
            "RA",
            "Dec",
            "Obs. Date Start UT",
            "Obs. Date Start",
            "Obs. Date End UT",
            "Obs. Date End",
            "Level of file",
            "Detector",
            "Processing version",
            "Full file path",
            "Filename",
            "Processing Date",
            "Delivery Date",
        ]
        df = self.to_pandas()
        df["lvlfilepath"] = df.lvldir + "/" + df.lvlfilename
        k = np.asarray([os.path.isfile(path) for path in df.lvlfilepath.values])
        if not k.any():
            logger.info(
                f"No files found for level {self.level} archive manifest. Storing at {manifest_path}"
            )
            return pd.DataFrame(columns=columns)
        adf = (
            df[k][
                [
                    "targ_id",
                    "targ_ra",
                    "targ_dec",
                    "jd",
                    "instrmnt",
                    "pfsoftver",
                    "lvlfilepath",
                    "exptime",
                ]
            ]
            .copy()
            .reset_index(drop=True)
        )
        adf.loc[:, "Obs. Date Start UT"] = Time(adf.jd.values, format="jd").isot
        adf.loc[:, "Obs. Date End"] = (
            adf["jd"].values.copy() + adf.exptime.values.copy() / 86400.0
        )
        adf.loc[:, "Obs. Date End UT"] = Time(
            adf["Obs. Date End"].values.copy(), format="jd"
        ).isot
        adf["Level of file"] = self.level
        adf["Processing Date"] = [
            datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
            .replace(tzinfo=None)
            .isoformat(timespec="milliseconds")
            for path in adf["lvlfilepath"]
        ]
        adf["Delivery Date"] = Time.now().isot
        adf["Filename"] = [path.split("/")[-1] for path in adf.lvlfilepath.values]
        adf = adf.rename(
            {
                "targ_id": "Target Name",
                "targ_ra": "RA",
                "targ_dec": "Dec",
                "jd": "Obs. Date Start",
                "instrmnt": "Detector",
                "pfsoftver": "Processing version",
                "lvlfilepath": "Full file path",
            },
            axis="columns",
        )
        adf = adf[columns]
        logger.info(f"Archive manifest stored at {manifest_path}")

        adf.to_csv(manifest_path, index=False)
        return adf
