# Standard library
import json

# Third-party
import numpy as np
import pandas as pd
from astropy.time import Time

# First-party/Local
from pandorafits.database import engineering
from pandorafits.database.engineering import JSONS, EngineeringDataBase
from pandorafits.database.engineering import PayloadDataBase


def write_export(folder, name, packets):
    """Telemetry export: one record per packet, values as {"val": ...}."""
    folder.mkdir(parents=True, exist_ok=True)
    records = [
        {"primaryKey": {"packetScTimeTai": time}}
        | {key: {"val": value} for key, value in values.items()}
        for time, values in packets
    ]
    (folder / name).write_text(json.dumps(records))


def payload_packets(day, cold_tip):
    # Columns in a different order than the table, plus an unknown one.
    return [
        (
            f"{day}T00:00:{37 + second:02d}.000",
            {
                "unknownField": 1,
                "pldCceTemperatureColdTip2": cold_tip + second,
                "pldPduErrorCode": second,
            },
        )
        for second in range(3)
    ]


def test_payload_database_ingests_every_day(tmp_path, monkeypatch):
    root = tmp_path / "engineering_data"
    write_export(
        root / "2026-09-05",
        "PayloadTelemetry.json",
        payload_packets("2026-09-05", 125.0),
    )
    write_export(
        root / "2026-09-06",
        "PayloadTelemetry.json",
        payload_packets("2026-09-06", 126.0),
    )
    # A second download of the same day and a staging folder.
    write_export(
        root / "later" / "2026-09-06",
        "PayloadTelemetry.json",
        payload_packets("2026-09-06", 126.0),
    )
    write_export(
        root / "staging" / "2026-09-07",
        "PayloadTelemetry.json",
        payload_packets("2026-09-07", 999.0),
    )
    monkeypatch.setattr(engineering, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        PayloadDataBase, "db_path", str(tmp_path / "payload.db")
    )

    with PayloadDataBase() as db:
        db.crawl_and_add()
        table = pd.read_sql_query("SELECT * FROM payload ORDER BY jd", db.conn)

    assert len(table) == 6
    # Packet times are TAI; the database stores UTC (37 s earlier in 2026).
    expected = Time("2026-09-05T00:00:00", scale="utc").jd
    assert np.isclose(table["jd"].iloc[0], expected, atol=1e-8)
    assert list(table["pldCceTemperatureColdTip2"]) == [
        125.0,
        126.0,
        127.0,
        126.0,
        127.0,
        128.0,
    ]
    assert list(table["pldPduErrorCode"]) == [0, 1, 2, 0, 1, 2]
    assert table["pldCceTemperatureColdTip1"].isna().all()


def test_stale_day_is_updated(tmp_path, monkeypatch):
    # A day within 15 days of ingestion is read again on the next crawl.
    day = Time.now().isot[:10]
    folder = tmp_path / "engineering_data" / day
    write_export(folder, "PayloadTelemetry.json", payload_packets(day, 125.0))
    monkeypatch.setattr(engineering, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        PayloadDataBase, "db_path", str(tmp_path / "payload.db")
    )
    with PayloadDataBase() as db:
        db.crawl_and_add()
    write_export(folder, "PayloadTelemetry.json", payload_packets(day, 130.0))
    with PayloadDataBase() as db:
        db.crawl_and_add()
        table = pd.read_sql_query("SELECT * FROM payload ORDER BY jd", db.conn)
    assert list(table["pldCceTemperatureColdTip2"]) == [130.0, 131.0, 132.0]


def test_engineering_database_joins_exports(tmp_path, monkeypatch):
    folder = tmp_path / "engineering_data" / "2026-09-05"
    columns = {
        "Analogs.json": "pdu2Heater1TempPayloadEbox",
        "Power.json": "pdu2Heater1Status",
        "AttCtrl.json": "positionError1",
        "Refs.json": "qEcefWrtEci1",
        "AttDet.json": "qBodyWrtEci1",
        "Sada.json": "sadaMode",
        "Tracker.json": "tracker1DataValid",
    }
    assert sorted(columns) == sorted(JSONS)
    for name, column in columns.items():
        write_export(
            folder,
            name,
            [
                ("2026-09-05T00:00:37.000", {column: 1}),
                ("2026-09-05T00:00:38.000", {column: 2}),
            ],
        )
    monkeypatch.setattr(engineering, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        EngineeringDataBase, "db_path", str(tmp_path / "engineering.db")
    )
    with EngineeringDataBase() as db:
        db.crawl_and_add()
        table = pd.read_sql_query(
            "SELECT * FROM engineering ORDER BY jd", db.conn
        )
    assert len(table) == 2
    for column in columns.values():
        assert list(table[column]) == [1, 2]
