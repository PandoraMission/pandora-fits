# Third-party
import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

# First-party/Local
from pandorafits import NIRDAReference, VISDAReference
from pandorafits.nirda import NIRDALevel1HDUList
from pandorafits.processing import ProcessingMixins, get_scene_flux


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


@pytest.mark.parametrize(
    "reference, column, magnitudes",
    [
        (NIRDAReference, "j_m", [9.0, None, np.nan, 11.0]),
        (VISDAReference, "phot_bp_mean_mag", [11.0, 12.0, None, 13.5]),
    ],
)
def test_scene_flux_uses_detector_band(reference, column, magnitudes):
    # pandoraaperture builds its catalog with object dtype columns.
    other = "phot_bp_mean_mag" if column == "j_m" else "j_m"
    cat = pd.DataFrame(
        {column: magnitudes, other: [1.0, 2.0, 3.0, 4.0]}, dtype=object
    )
    flux, used_column, n_missing = get_scene_flux(cat, reference)
    assert used_column == column
    assert flux.unit == "ct / s"
    has_magnitude = np.array([m is not None and m == m for m in magnitudes])
    assert n_missing == (~has_magnitude).sum()
    assert np.all(flux.value[~has_magnitude] == 0)
    expected = reference.magnitude_to_flux(
        np.array(magnitudes, dtype=float)[has_magnitude]
    )
    assert np.allclose(flux.value[has_magnitude], expected)


def make_pointed_nirda_hdulist(ra, astrometry_header=None):
    header = fits.Header(
        [
            ("INSTRMNT", "NIRDA"),
            ("CORSTIME", 842115335),
            ("FINETIME", 0),
            ("FRMTIME", 237),
            ("FRMSTOT", 100),
            ("RA", ra),
            ("DEC", 20.0),
            ("ROLL", 30.0),
            ("TARG_RA", 10.0),
            ("TARG_DEC", 20.0),
        ]
    )
    hdus = [
        fits.PrimaryHDU(header=header),
        fits.ImageHDU(np.zeros((2, 4, 4)), name="SCIENCE"),
    ]
    if astrometry_header is not None:
        hdus.append(fits.BinTableHDU(header=astrometry_header))
    return NIRDALevel1HDUList(fits.HDUList(hdus), validate=False)


def test_append_wcs_replaces_earlier_wcs():
    hdulist = make_pointed_nirda_hdulist(ra=10.0)
    hdulist._append_wcs()
    hdulist[0].header["RA"] = 10.01
    hdulist._append_wcs()

    header = hdulist["SCIENCE"].header
    assert sum(card.keyword == "CRVAL1" for card in header.cards) == 1
    expected = NIRDAReference.get_wcs_from_VITL(10.01, 20.0, 30.0)
    assert np.isclose(header["CRVAL1"], expected.wcs.crval[0])
    # POSX, the stored cards and the WCS read back all agree.
    posx = hdulist.wcs.world_to_pixel(hdulist.coord)[0]
    assert np.isclose(hdulist[0].header["POSX"], posx)
    assert np.isclose(hdulist.wcs.wcs.crval[0], header["CRVAL1"])


class FakeAstrometryDataBase:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def to_pandas(self, time_range, columns):
        return pd.DataFrame({column: np.ones(3) for column in columns}).assign(
            jd=np.linspace(*time_range, 3)
        )


def test_astrometry_extension_keeps_solution_cards(monkeypatch):
    monkeypatch.setattr(
        "pandorafits.processing.AstrometryDataBase", FakeAstrometryDataBase
    )
    old = fits.Header(
        [
            ("EXTNAME", "ASTROMETRY"),
            ("ASCAL", 0.7875, "Plate Scale for ast soln"),
            ("CHECKSUM", "stalestalestale0"),
            ("DATASUM", "0"),
        ]
    )
    hdulist = make_pointed_nirda_hdulist(ra=10.0, astrometry_header=old)
    hdulist._update_astrometry_extension()

    header = hdulist["ASTROMETRY"].header
    keywords = [card.keyword for card in header.cards]
    assert keywords.count("EXTNAME") == 1
    assert header["ASCAL"] == 0.7875
    assert "CHECKSUM" not in keywords and "DATASUM" not in keywords
    assert len(hdulist["ASTROMETRY"].data) == 3
