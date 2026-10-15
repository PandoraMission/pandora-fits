# Third-party
import pytest

# First-party/Local
from pandorafits import logger
from pandorafits.nirda import NIRDALevel0HDUList
from pandorafits.visda import VISDAFFILevel0HDUList, VISDALevel0HDUList


@pytest.mark.parametrize(
    "hdulist_class, filename",
    [
        (NIRDALevel0HDUList, "nirda-level0.fits"),
        (VISDALevel0HDUList, "visda-level0.fits"),
        (VISDAFFILevel0HDUList, "visda-ffi-level0.fits"),
    ],
)
def test_create_dummy_data(hdulist_class, filename, tmp_path):
    level = logger.level
    logger.setLevel("ERROR")
    path = tmp_path / filename
    hdulist_class().writeto(str(path), overwrite=True)
    logger.setLevel(level)
    assert path.is_file()
