"""Tools for generating fallback database of SOC targets"""

import warnings
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path

import pandas as pd
from astropy.time import Time

from .. import CALENDAR_DIR, LEVEL0_DIR
from ..utils import get_dpc_hashkey
from .mixins import DataBaseMixins


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
            timing = op.find("pandora:Timing", namespace)
            start = Time(timing.find("pandora:Start", namespace).text, format="isot").jd
            end = Time(timing.find("pandora:Start", namespace).text, format="isot").jd
            priority = op.find("pandora:Priority", namespace).text
            bs = op.find("pandora:Boresight", namespace)
            boresight_ra = bs.find("pandora:RA", namespace).text
            boresight_dec = bs.find("pandora:DEC", namespace).text
            roll_elem = bs.find("pandora:Roll", namespace)
            boresight_roll = roll_elem.text if roll_elem is not None else np.nan
            params = seq_elem.find("pandora:Payload_Parameters", namespace)
            vis_params = params.find("pandora:AcquireVisCamScienceData", namespace)
            targ_id = vis_params.find("pandora:TargetID", namespace).text
            targ_ra = vis_params.find("pandora:TargetRA", namespace).text
            targ_dec = vis_params.find("pandora:TargetDEC", namespace).text

            dfs.append(
                pd.DataFrame(
                    [
                        start,
                        end,
                        Time(metadata["created"]).jd,
                        targ_id,
                        boresight_ra,
                        boresight_dec,
                        boresight_roll,
                        targ_ra,
                        targ_dec,
                        priority,
                    ]
                ).T
            )
    df = pd.concat(dfs)
    df["visit_start"] = df.groupby(2)[0].transform("min")
    df["visit_end"] = df.groupby(2)[0].transform("max")
    df = df.rename(
        {
            0: "start",
            1: "end",
            2: "created",
            3: "targ_id",
            4: "boresight_ra",
            5: "boresight_dec",
            6: "boresight_roll",
            7: "targ_ra",
            8: "targ_dec",
            9: "priority",
        },
        axis="columns",
    ).reset_index(drop=True)

    return df.drop_duplicates(
        [
            "visit_start",
            "visit_end",
            "created",
            "targ_id",
            "boresight_ra",
            "boresight_dec",
            "boresight_roll",
        ]
    )


class TargetDataBase(DataBaseMixins):
    """Database for managing astrometry of Pandora"""

    table_name = "targets"
    db_path = f"{LEVEL0_DIR}/targets.db"
    _sql_key_dict = {
        "filename": "TEXT",
        "visit_start": "FLOAT",
        "visit_end": "FLOAT",
        "created": "FLOAT",
        "targ_id": "TEXT",
        "boresight_ra": "FLOAT",
        "boresight_dec": "FLOAT",
        "boresight_roll": "FLOAT",
        "targ_ra": "FLOAT",
        "targ_dec": "FLOAT",
        "priority": "INT",
        "dpc_hash_key": "TEXT",
    }

    def __repr__(self):
        return "Pandora TargetDataBase"

    def crawl_and_add(self):
        root = CALENDAR_DIR
        for cal_type in ["PAN-SCICAL-TST"]:
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
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")  # capture all warnings

            df = calendar_to_targets(filename)
            return [
                (
                    filename.split("/")[-1],
                    float(df.iloc[idx].visit_start),
                    float(df.iloc[idx].visit_end),
                    float(df.iloc[idx].created),
                    df.iloc[idx].targ_id,
                    float(df.iloc[idx].boresight_ra),
                    float(df.iloc[idx].boresight_dec),
                    float(df.iloc[idx].boresight_roll),
                    float(df.iloc[idx].targ_ra),
                    float(df.iloc[idx].targ_dec),
                    int(df.iloc[idx].priority),
                    get_dpc_hashkey(
                        df.iloc[idx].targ_id,
                        float(df.iloc[idx].targ_ra),
                        float(df.iloc[idx].targ_dec),
                    ),
                )
                for idx in range(len(df))
            ]

    def add_target(
        self,
        created,
        visit_start,
        visit_end,
        targ_id,
        boresight_ra,
        boresight_dec,
        boresight_roll,
        targ_ra,
        targ_dec,
        priority,
    ):
        sql = """
            INSERT INTO targets (filename, created, visit_start, visit_end, targ_id, boresight_ra, boresight_dec, boresight_roll, targ_ra, targ_dec, priority, dpc_hash_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

        values = (
            "user",
            Time(created).jd,
            Time(visit_start).jd,
            Time(visit_end).jd,
            str(targ_id),
            float(boresight_ra),
            float(boresight_dec),
            float(boresight_roll),
            float(targ_ra),
            float(targ_dec),
            int(priority),
            get_dpc_hashkey(str(targ_id), float(targ_ra), float(targ_dec)),
        )

        self.conn.execute(sql, values)
        self.conn.commit()
