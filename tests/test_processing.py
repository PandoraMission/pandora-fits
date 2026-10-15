# Third-party
import numpy as np
from astropy.io import fits

# First-party/Local
from pandorafits.nirda import NIRDALevel1HDUList
from pandorafits.processing import ProcessingMixins


class OnesReferenceHDUList(ProcessingMixins, fits.HDUList):
    """HDUList whose reference detector images are all ones."""

    def _get_reference_detector_image(self, name):
        return np.ones((2, 2))


def test_detector_image_is_applied_once():
    hdulist = OnesReferenceHDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(np.full((1, 2, 2), 5.0), name="SCIENCE"),
        ]
    )
    hdulist._subtract_bias()
    assert np.all(hdulist["SCIENCE"].data == 4)
    # A second bias subtraction is refused, other steps still apply.
    hdulist._subtract_bias()
    assert np.all(hdulist["SCIENCE"].data == 4)
    hdulist._subtract_dark()
    assert np.all(hdulist["SCIENCE"].data == 3)
    assert list(hdulist[0].header["COMMENT"]) == [
        "Applied bias.",
        "Applied dark.",
    ]


def make_nirda_hdulist(posy, vectors=False):
    header = fits.Header(
        [("ROISTRTY", 959), ("ROISIZEY", 256), ("POSY", posy)]
    )
    hdus = [fits.PrimaryHDU(header=header)]
    if vectors:
        hdus.append(
            fits.BinTableHDU.from_columns(
                [fits.Column("posy", "D", array=np.full(5, posy))],
                name="VECTORS",
            )
        )
    return NIRDALevel1HDUList(fits.HDUList(hdus), validate=False)


def test_wavelength_solution_without_vectors():
    posy = 1086.4
    with_vectors = make_nirda_hdulist(posy, vectors=True)
    with_vectors._append_wavelength()
    from_header = make_nirda_hdulist(posy)
    from_header._append_wavelength()

    table = from_header["WAVELENGTH_SOLUTION"].data
    assert np.allclose(table["drow"], np.arange(959, 959 + 256) - posy)
    for column in ["drow", "wavelength", "sensitivity"]:
        assert np.allclose(
            table[column],
            with_vectors["WAVELENGTH_SOLUTION"].data[column],
            equal_nan=True,
        )
