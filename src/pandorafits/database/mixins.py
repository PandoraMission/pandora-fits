"""Mixins for databases"""

import numpy as np
import pandas as pd
from astropy.time import Time
import threading
import sqlite3


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
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()

    def to_pandas(self, time_range=None, targ_id=None, dpc_obs_id=None):
        sql = "SELECT * FROM pointings"
        params = []

        where_clauses = []

        if time_range is not None:
            start, end = _process_time(time_range[0]), _process_time(time_range[1])
            start, end = np.sort([start, end])
            where_clauses.append("jd BETWEEN ? AND ?")
            params.extend([start, end])

        if targ_id is not None:
            where_clauses.append("targ_id = ?")
            params.append(targ_id)

        if dpc_obs_id is not None:
            where_clauses.append("dpc_obs_id = ?")
            params.append(dpc_obs_id)

        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        return pd.read_sql_query(sql, self.conn, params=params)

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
