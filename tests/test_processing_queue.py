# Standard library
import sqlite3

# Third-party
import pytest

# First-party/Local
from pandorafits.database import level1, levele
from pandorafits.database.level1 import Level1DataBase
from pandorafits.database.levele import LevelEDataBase
from pandorafits.database.mixins import file_chunk


def make_level0(folder, nfiles=20, bad=()):
    """A minimal Level 0 database: file names, folders, checksums, times."""
    connection = sqlite3.connect(folder / "level0.db")
    connection.execute(
        "CREATE TABLE pointings (filename TEXT PRIMARY KEY, "
        "lvlfilename TEXT, lvldir TEXT, jd FLOAT, "
        "badchecksum INT, baddatasum INT)"
    )
    names = [f"file{index:02d}_InfImg.fits" for index in range(nfiles)]
    connection.executemany(
        "INSERT INTO pointings VALUES (?, ?, ?, ?, ?, ?)",
        [
            (name, name, "/data", 2461300.0, int(name in bad), 0)
            for name in names
        ],
    )
    connection.commit()
    connection.close()
    return names


@pytest.fixture
def level1_database(tmp_path, monkeypatch):
    monkeypatch.setattr(level1, "LEVEL0_DIR", str(tmp_path))
    monkeypatch.setattr(Level1DataBase, "db_path", str(tmp_path / "level1.db"))
    return tmp_path


def test_chunks_cover_every_file_once(level1_database):
    names = make_level0(level1_database, bad=["file03_InfImg.fits"])
    with Level1DataBase() as db:
        assert db.n_files_to_process == 19
        chunks = [
            [row[0] for row in db.get_x_to_process("src.filename", 4, chunk)]
            for chunk in range(4)
        ]
    selected = sorted(name for chunk in chunks for name in chunk)
    assert selected == sorted(set(names) - {"file03_InfImg.fits"})
    for chunk, members in enumerate(chunks):
        assert all(file_chunk(name, 4) == chunk for name in members)


def test_chunk_keeps_its_files_as_others_finish(level1_database):
    make_level0(level1_database)
    with Level1DataBase() as db:
        before = db.get_x_to_process("src.filename", 3, 1)
        # Another chunk's files are processed in the meantime.
        for (name,) in db.get_x_to_process("src.filename", 3, 0):
            db.cur.execute(
                "INSERT INTO pointings (filename) VALUES (?)", (name,)
            )
        db.conn.commit()
        assert db.get_x_to_process("src.filename", 3, 1) == before


@pytest.fixture
def levele_database(tmp_path, monkeypatch):
    monkeypatch.setattr(levele, "LEVEL0_DIR", str(tmp_path))
    monkeypatch.setattr(LevelEDataBase, "db_path", str(tmp_path / "levele.db"))
    return tmp_path


def test_levele_selects_new_and_stale_files(levele_database):
    names = make_level0(levele_database, nfiles=4, bad=["file03_InfImg.fits"])
    with LevelEDataBase() as db:
        # file01: made 5 days after the observation (stale); file02: 30
        # days (final); file03: stale but with a bad checksum.
        for name, days in [(names[1], 5), (names[2], 30), (names[3], 5)]:
            db.add_entry(
                (name, name, "/data", "/out", 2461300.0, 2461300.0 + days)
            )
        selected = [row[0] for row in db.get_x_to_process("src.filename")]
        assert selected == [names[0], names[1]]
        assert db.n_files_to_process == 2


def test_levele_remade_file_replaces_its_entry(levele_database):
    names = make_level0(levele_database, nfiles=1)
    with LevelEDataBase() as db:
        db.add_entry(
            (names[0], names[0], "/data", "/out", 2461300.0, 2461305.0)
        )
        db.add_entry(
            (names[0], names[0], "/data", "/out", 2461300.0, 2461320.0)
        )
        db.cur.execute("SELECT create_time FROM pointings")
        assert db.cur.fetchall() == [(2461320.0,)]
        assert db.n_files_to_process == 0
