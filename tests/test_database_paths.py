# Standard library
from pathlib import Path

# Third-party
import numpy as np
from astropy.io import fits

# First-party/Local
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
