# Standard library
import os
import sqlite3
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


def get_engineering_entry(filename):
    with warnings.catch_warnings(record=True) as w:  # noqa
        warnings.simplefilter("always")  # capture all warnings
        dfs = []
        for jsonname in JSONS:
            df = pd.read_json(filename + f"/{jsonname}")
            df = df.map(
                lambda x: x["val"] if isinstance(x, dict) and "val" in x else x
            )
            df["JD"] = df["primaryKey"].map(
                lambda x: (
                    Time(x["packetScTimeTai"], format="isot").jd
                    if isinstance(x, dict) and "packetScTimeTai" in x
                    else x
                )
            )
            df.drop("primaryKey", axis="columns", inplace=True)
            df.set_index("JD", inplace=True)
            dfs.append(df)
        df = pd.concat(dfs, axis=1)
        df = df.reset_index()
        df.drop_duplicates("JD", keep="first", inplace=True)
        return [i[1].values for i in df.iterrows()]


def get_payload_entry(filename):
    with warnings.catch_warnings(record=True) as w:  # noqa
        warnings.simplefilter("always")  # capture all warnings
        df = pd.read_json(filename)
        df = df.map(
            lambda x: x["val"] if isinstance(x, dict) and "val" in x else x
        )
        df["JD"] = df["primaryKey"].map(
            lambda x: (
                Time(x["packetScTimeTai"], format="isot").jd
                if isinstance(x, dict) and "packetScTimeTai" in x
                else x
            )
        )
        df.drop("primaryKey", axis="columns", inplace=True)
        df.set_index("JD", inplace=True)
        df = df.reset_index()
        df.drop_duplicates("JD", keep="first", inplace=True)
        return [i[1].values for i in df.iterrows()]


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
        key_string = ", ".join(
            [f"{key}" for key, item in self._sql_key_dict.items()]
        )
        value_string = ", ".join(["?"] * len(self._sql_key_dict))
        self.update_str = f"""INSERT IN)TO {self.table_name} ({key_string}) VALUES ({value_string}) ON CONFLICT (JD) DO NOTHING"""
        self.conn = sqlite3.connect(self.db_path)
        self.cur = self.conn.cursor()
        self.cur.execute(
            """
            CREATE TABLE IF NOT EXISTS completed (eng_day FLOAT PRIMARY KEY, updated_day FLOAT)
            """
        )
        self.conn.commit()

    def crawl_and_add(self):
        root = DATA_DIR + "/engineering_data/"
        paths = np.sort(
            [str(path) for path in Path(root).rglob("Analogs.json")]
        )
        path_idxs = np.unique(
            [p.split("/")[-1] for p in paths[::-1]], return_index=True
        )[1]
        paths = np.sort(paths[::-1][path_idxs])
        with logging_redirect_tqdm():
            for path in tqdm(paths, desc="Engineering files"):
                if not self.check_if_completed(path):
                    rows = get_engineering_entry(
                        "/".join(path.split("/")[:-1])
                    )
                    self.add_entries(rows)
                    self.update_completed(path)

    def check_if_completed(self, path):
        t_eng = Time(path.split("/")[-2], format="isot").jd
        sql = f"SELECT updated_day FROM completed WHERE eng_day = {t_eng}"
        df = pd.read_sql_query(sql, self.conn)
        if len(df) > 0:
            if (df.updated_day.values - t_eng) < 15:
                logger.info(f"{path} may be stale, rewriting to database.")
                # Delete the line
                self.conn.execute(
                    f"DELETE FROM completed WHERE eng_day = {t_eng}",
                )
                self.conn.commit()
                return False
            else:
                logger.info(f"{path} already in database.")
                return True
        elif len(df) == 0:
            return False

    def update_completed(self, path):
        update_str = f"""INSERT INTO completed (eng_day, updated_day) VALUES ({Time(path.split("/")[-2], format="isot").jd}, {Time.now().jd})"""  # noqa
        self.cur.execute(update_str)
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
        root = DATA_DIR + "/engineering_data/"
        paths = np.sort(
            [str(path) for path in Path(root).rglob("PayloadTelemetry.json")]
        )
        path_idxs = np.unique(
            [p.split("/")[-1] for p in paths[::-1]], return_index=True
        )[1]
        paths = np.sort(paths[::-1][path_idxs])
        with logging_redirect_tqdm():
            for path in tqdm(paths, desc="Payload files"):
                if not self.check_if_completed(path):
                    rows = get_payload_entry(path)
                    self.add_entries(rows)
                    self.update_completed(path)
