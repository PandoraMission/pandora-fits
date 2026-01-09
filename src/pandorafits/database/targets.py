"""Tools for generating fallback database of SOC targets"""

import pandas as pd
from lxml import etree
import xml.etree.ElementTree as ET
from astropy.time import Time
import os
import warnings
from datetime import timedelta
from pathlib import Path

from astropy.io import fits
from astropy.time import Time

from .. import DATA_DIR, __version__, LEVEL0_DIR
from ..utils import get_dpc_hashkey

import os
import warnings
from datetime import timedelta
from pathlib import Path

import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time

from .. import DATA_DIR, CALENDAR_DIR, __version__
from ..roll import get_roll
from .mixins import DataBaseMixins
import numpy as np


def calendar_to_targets(fname):
    tree = ET.parse(fname)
    root = tree.getroot()

    # Define the namespace
    namespace = {"pandora": "/pandora/calendar/"}

    # Parse metadata using namespace
    meta = root.find("pandora:Meta", namespace)

    if meta is not None:
        metadata = {
            "valid_from": meta.get("Valid_From"),
            "expires": meta.get("Expires"),
            "calendar_weights": meta.get("Calendar_Weights"),
            "ephemeris": meta.get("Ephemeris"),
            "keepout_angles": meta.get("Keepout_Angles"),
            "observation_sequence_duration": meta.get(
                "Observation_Sequence_Duration_hrs"
            ),
            "removed_sequences_shorter_than": meta.get(
                "Removed_Sequences_Shorter_Than_min"
            ),
            "created": meta.get("Created"),
            "delivery_id": meta.get("Delivery_Id"),
        }
    else:
        metadata = {}

    # Parse visits using namespace
    dfs = []

    visit_elements = root.findall("pandora:Visit", namespace)
    for visit_elem in visit_elements:
        seq_elements = visit_elem.findall("pandora:Observation_Sequence", namespace)
        for seq_elem in seq_elements:
            op = seq_elem.find("pandora:Observational_Parameters", namespace)
            targ_id = op.find("pandora:Target", namespace).text
            bs = op.find("pandora:Boresight", namespace)
            targ_ra = bs.find("pandora:RA", namespace).text
            targ_dec = bs.find("pandora:RA", namespace).text
            dfs.append(
                pd.DataFrame(
                    [Time(metadata["created"]).jd, targ_id, targ_ra, targ_dec]
                ).T
            )
    df = (
        pd.concat(dfs)
        .drop_duplicates()
        .rename(
            {0: "created", 1: "targ_id", 2: "targ_ra", 3: "targ_dec"}, axis="columns"
        )
        .reset_index(drop=True)
    )
    return df


class TargetDataBase(DataBaseMixins):
    """Database for managing astrometry of Pandora"""

    table_name = "targets"
    db_path = f"{LEVEL0_DIR}/targets.db"
    _sql_key_dict = {
        "filename": "TEXT",
        "created": "FLOAT",
        "targ_id": "TEXT",
        "targ_ra": "FLOAT",
        "targ_dec": "FLOAT",
        "dpc_hash_key": "TEXT",
    }

    def __repr__(self):
        return "Pandora TargetDataBase"

    def crawl_and_add(self):
        root = CALENDAR_DIR
        for cal_type in ["PAN-LONGCAL-TST", "PAN-SCICAL-TST"]:
            paths = [
                str(path)
                for path in Path(root).rglob(f"*{cal_type}*.xml")
                if not self.check_filename_in_database(str(path))
            ]
            rows = []
            for path in paths:
                values = self.get_entry(path)
                if values is not None:
                    [rows.append(v) for v in values]
            self.add_entries(rows)

    def get_entry(self, filename):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # capture all warnings

            df = calendar_to_targets(filename)
            return [
                (
                    filename.split("/")[-1],
                    float(df.iloc[idx].created),
                    df.iloc[idx].targ_id,
                    float(df.iloc[idx].targ_ra),
                    float(df.iloc[idx].targ_dec),
                    get_dpc_hashkey(
                        df.iloc[idx].targ_id,
                        float(df.iloc[idx].targ_ra),
                        float(df.iloc[idx].targ_dec),
                    ),
                )
                for idx in range(len(df))
            ]

    def add_target(self, targ_id, targ_ra, targ_dec):
        sql = """
            INSERT INTO targets (filename, created, targ_id, targ_ra, targ_dec, dpc_hash_key)
            VALUES (?, ?, ?, ?, ?, ?)
            """

        values = (
            "user",
            Time("1991-07-25 12:00:00").jd,
            str(targ_id),
            float(targ_ra),
            float(targ_dec),
            get_dpc_hashkey(str(targ_id), float(targ_ra), float(targ_dec)),
        )

        self.conn.execute(sql, values)
        self.conn.commit()
