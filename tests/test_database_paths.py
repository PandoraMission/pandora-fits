# Standard library
from pathlib import Path

# Third-party
import numpy as np
from astropy.io import fits

# First-party/Local
from pandorafits.database import is_banned
from pandorafits.database.level0 import get_entry
from pandorafits.utils import split_path, unique_by_name


def test_split_path_on_any_platform(tmp_path):
    path = tmp_path / "data" / "2026-09-05__22-56-18_InfImg_TEST.fits"
    name, directory = split_path(str(path))
    assert name == path.name
    assert directory == (tmp_path / "data").as_posix()
    assert "\\" not in directory


def test_unique_by_name_keeps_one_path_per_file(tmp_path):
    # The same file downloaded to two folders, and one other file.
    first = str(tmp_path / "a" / "file1.fits")
    second = str(tmp_path / "b" / "file1.fits")
    other = str(tmp_path / "a" / "file2.fits")
    paths = unique_by_name([first, other, second])
    assert sorted(Path(p).name for p in paths) == ["file1.fits", "file2.fits"]
    # The path that sorts last is kept for a repeated name.
    assert second in paths and first not in paths
    assert list(paths) == sorted(paths)


def test_level0_entry_stores_name_and_directory(tmp_path):
    header = fits.Header(
        [
            ("INSTRMNT", "NIRDA"),
            ("CORSTIME", 842115335),
            ("FINETIME", 0),
            ("CRSOFTV", "v3.06"),
            ("FRMTIME", 237),
            ("FRMSTOT", 100),
            ("ROISIZEX", 4),
            ("ROISIZEY", 3),
            ("ROISTRTX", 1737),
            ("ROISTRTY", 959),
        ]
    )
    folder = tmp_path / "pandora_20260905T225600_TEST"
    folder.mkdir()
    path = folder / "2026-09-05__22-56-18_InfImg_TEST.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(header=header),
            fits.ImageHDU(np.zeros((2, 3, 4), np.int16), name="SCIENCE"),
        ]
    ).writeto(path)

    entry = get_entry(str(path))
    assert entry[0] == entry[1] == path.name
    assert entry[2] == entry[3] == folder.as_posix()


def test_banned_files_and_folders(tmp_path, monkeypatch):
    data = tmp_path / "staging_area" / "data"
    monkeypatch.setattr("pandorafits.database.DATA_DIR", str(data))
    good = "2026-09-05__22-56-18_InfImg_TEST.fits"
    assert not is_banned(data / "pandora_20260905" / good)
    # Banned file names and banned folders below the data directory.
    assert is_banned(data / "pandora_20250905" / "2025-09-05_InfImg.fits")
    assert is_banned(data / "staging" / good)
    assert is_banned(data / "engineering_data" / "2026-09-05" / good)
    # The data directory's own path ("staging_area") and folder names
    # that look like banned file-name strings do not ban a file.
    assert not is_banned(data / "2026-07-01" / good)
    # Outside the data directory every folder is checked.
    assert is_banned(tmp_path / "staging" / good)
