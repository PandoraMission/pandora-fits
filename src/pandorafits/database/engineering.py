# Standard library
import os
import warnings
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
from astropy.time import Time
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# First-party/Local
from pandorafits import DATA_DIR, LEVEL0_DIR
from .. import logger
from .mixins import DataBaseMixins

JSONS = [
    "Analogs.json",
    "Power.json",
    "AttCtrl.json",
    "Refs.json",
    "AttDet.json",
    "Sada.json",
    "Tracker.json",
]


def _read_telemetry_json(filename):
    """One telemetry JSON export as a table indexed by JD (UTC).

    Values stored as ``{"val": ...}`` are unpacked. Packet times
    (``packetScTimeTai``) are in TAI and converted to UTC to match the other
    databases; packets without a time are dropped, and repeated times keep
    their first packet.
    """
    df = pd.read_json(filename)
    df = df.map(
        lambda x: x["val"] if isinstance(x, dict) and "val" in x else x
    )
    tai = df["primaryKey"].map(
        lambda x: (
            x["packetScTimeTai"]
            if isinstance(x, dict) and "packetScTimeTai" in x
            else None
        )
    )
    df = df[tai.notna().values].drop(columns="primaryKey")
    df.index = Time(list(tai[tai.notna()]), format="isot", scale="tai").utc.jd
    df.index.name = "jd"
    return df[~df.index.duplicated(keep="first")]


def get_engineering_entry(day_folder):
    """Engineering telemetry of one day folder, one row per packet time."""
    with warnings.catch_warnings(record=True) as w:  # noqa
        warnings.simplefilter("always")  # capture all warnings
        dfs = [_read_telemetry_json(Path(day_folder) / name) for name in JSONS]
        return pd.concat(dfs, axis=1).reset_index()


def get_payload_entry(filename):
    """Payload telemetry of one JSON export, one row per packet time."""
    with warnings.catch_warnings(record=True) as w:  # noqa
        warnings.simplefilter("always")  # capture all warnings
        return _read_telemetry_json(filename).reset_index()


def _sql_value(value):
    """A value sqlite3 can store: missing values as NULL, NumPy scalars as
    Python scalars (sqlite3 cannot store NumPy integers)."""
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _day_files(json_name):
    """The telemetry export of each day folder below
    ``DATA_DIR/engineering_data``, skipping staging folders.

    A day can be downloaded more than once; the path that sorts last is kept.
    """
    root = Path(DATA_DIR) / "engineering_data"
    days = {}
    for path in sorted(root.rglob(json_name)):
        folders = path.relative_to(root).parts[:-1]
        if any("staging" in folder for folder in folders):
            continue
        days[path.parent.name] = path
    return [days[day] for day in sorted(days)]


class EngineeringDataBase(DataBaseMixins):
    """Database for managing engineering data of Pandora"""

    table_name = "engineering"
    db_path = os.path.join(LEVEL0_DIR, "engineering.db")
    _sql_key_dict = {
        "jd": "FLOAT PRIMARY KEY",
        "pdu2Heater1TempPayloadEbox": "FLOAT",
        "pdu2Heater2TempPayloadObaHeatPipe2": "FLOAT",
        "pdu2Heater3TempPayloadObaHeatPipe3": "FLOAT",
        "pdu2Heater4TempPayloadHeatPipeThermalBlock": "FLOAT",
        "pdu2Heater5TempPayloadCss": "FLOAT",
        "pdu2Heater11TempObaHighPwrTrim2": "FLOAT",
        "pdu2Heater12TempObaHighPwrTrim3": "FLOAT",
        "pdu2Heater13TempPlRadiatorTrim": "FLOAT",
        "pdu2Heater14TempIrDetector": "FLOAT",
        "pdu2Heater15TempVisDetector": "FLOAT",
        "pdu2Heater15Status": "INT",
        "pdu2Heater14Status": "INT",
        "pdu2Heater13Status": "INT",
        "pdu2Heater12Status": "INT",
        "pdu2Heater11Status": "INT",
        "pdu2Heater5Status": "INT",
        "pdu2Heater4Status": "INT",
        "pdu2Heater3Status": "INT",
        "pdu2Heater2Status": "INT",
        "pdu2Heater1Status": "INT",
        "positionError1": "FLOAT",
        "positionError2": "FLOAT",
        "payloadError1": "FLOAT",
        "payloadError2": "FLOAT",
        "payloadError3": "FLOAT",
        "controlToPayload": "INT",
        "payloadControlUsed": "INT",
        "qEcefWrtEci1": "FLOAT",
        "qEcefWrtEci2": "FLOAT",
        "qEcefWrtEci3": "FLOAT",
        "qEcefWrtEci4": "FLOAT",
        "positionWrtEci1": "FLOAT",
        "positionWrtEci2": "FLOAT",
        "positionWrtEci3": "FLOAT",
        "velocityWrtEci1": "FLOAT",
        "velocityWrtEci2": "FLOAT",
        "velocityWrtEci3": "FLOAT",
        "qBodyWrtEci1": "FLOAT",
        "qBodyWrtEci2": "FLOAT",
        "qBodyWrtEci3": "FLOAT",
        "qBodyWrtEci4": "FLOAT",
        "residual1": "FLOAT",
        "residual2": "FLOAT",
        "residual3": "FLOAT",
        "tracker2DataValid": "INT",
        "tracker1DataValid": "INT",
        "measAttValid": "INT",
        "attitudeValid": "INT",
        "sadaMode": "FLOAT",
        "attStatus": "INT",
    }

    def __repr__(self):
        return "Pandora EngineeringDataBase"

    def __init__(self):
        super().__init__()
        columns = list(self._sql_key_dict)
        # A stale day is ingested again, so its packets replace stored ones.
        updates = ", ".join(f"{key} = excluded.{key}" for key in columns[1:])
        self.update_str = (
            f"INSERT INTO {self.table_name} ({', '.join(columns)}) "
            f"VALUES ({', '.join(['?'] * len(columns))}) "
            f"ON CONFLICT (jd) DO UPDATE SET {updates}"
        )
        self.cur.execute(
            """
            CREATE TABLE IF NOT EXISTS completed (eng_day FLOAT PRIMARY KEY, updated_day FLOAT)
            """
        )
        self.conn.commit()

    def _rows(self, df):
        """Table rows in this database's column order, as Python values."""
        df = df.reindex(columns=list(self._sql_key_dict))
        return [
            tuple(_sql_value(value) for value in row)
            for row in df.itertuples(index=False)
        ]

    def crawl_and_add(self):
        with logging_redirect_tqdm():
            for path in tqdm(
                _day_files("Analogs.json"), desc="Engineering files"
            ):
                if not self.check_if_completed(path):
                    self.add_entries(
                        self._rows(get_engineering_entry(path.parent))
                    )
                    self.update_completed(path)

    @staticmethod
    def _day_jd(path):
        """JD of the day folder holding a telemetry export."""
        return Time(Path(path).parent.name, format="isot").jd

    def check_if_completed(self, path):
        t_eng = self._day_jd(path)
        df = pd.read_sql_query(
            "SELECT updated_day FROM completed WHERE eng_day = ?",
            self.conn,
            params=(t_eng,),
        )
        if len(df) > 0:
            if (df.updated_day.values - t_eng) < 15:
                logger.info(f"{path} may be stale, rewriting to database.")
                # Delete the line
                self.conn.execute(
                    "DELETE FROM completed WHERE eng_day = ?", (t_eng,)
                )
                self.conn.commit()
                return False
            else:
                logger.info(f"{path} already in database.")
                return True
        elif len(df) == 0:
            return False

    def update_completed(self, path):
        self.cur.execute(
            "INSERT INTO completed (eng_day, updated_day) VALUES (?, ?)",
            (self._day_jd(path), Time.now().jd),
        )
        self.conn.commit()


class PayloadDataBase(EngineeringDataBase):
    """Database for managing payload engineering data of Pandora"""

    table_name = "payload"
    db_path = os.path.join(LEVEL0_DIR, "payload.db")
    _sql_key_dict = {
        "jd": "FLOAT PRIMARY KEY",
        "pldPcoImgSequenceInProgress": "INT",
        "pldH2rgImgSequenceInProgress": "INT",
        "pldPcoSensorTemp": "FLOAT",
        "pldPcoCameraTemp": "FLOAT",
        "pldPcoSupplyTemp": "FLOAT",
        "pldTcuAd590temperature": "FLOAT",
        "pldAd590temperatures0": "FLOAT",
        "pldAd590temperatures1": "FLOAT",
        "pldAd590temperatures2": "FLOAT",
        "pldAd590temperatures3": "FLOAT",
        "pldAd590temperatures4": "FLOAT",
        "pldAd590temperatures5": "FLOAT",
        "pldAd590temperatures6": "FLOAT",
        "pldAd590temperatures7": "FLOAT",
        "pldAd590temperatures8": "FLOAT",
        "pldAd590temperatures9": "FLOAT",
        "pldAd590temperatures10": "FLOAT",
        "pldAd590temperatures11": "FLOAT",
        "pldCurrent50vPolr0": "FLOAT",
        "pldCurrent50vPolr1": "FLOAT",
        "pldCurrent50vPolr5": "FLOAT",
        "pldCurrent280vTrunk": "FLOAT",
        "pldPduErrorCode": "INT",
        "pldTcuErrorCode": "INT",
        "pldCceMotorControlSmStatus": "INT",
        "pldCceTemperatureCompressorReject": "FLOAT",
        "pldCceVoltageMotor1": "INT",
        "pldCceVoltageMotor2": "INT",
        "pldCceTemperatureColdTip1": "FLOAT",
        "pldCceTemperatureColdTip2": "FLOAT",
    }

    def __repr__(self):
        return "Pandora PayloadDataBase"

    def crawl_and_add(self):
        with logging_redirect_tqdm():
            for path in tqdm(
                _day_files("PayloadTelemetry.json"), desc="Payload files"
            ):
                if not self.check_if_completed(path):
                    self.add_entries(self._rows(get_payload_entry(path)))
                    self.update_completed(path)
