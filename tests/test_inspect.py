# Third-party
import numpy as np
from astropy.io import fits

# First-party/Local
from pandorafits.inspect import SNRInspector, time_within_fraction
from pandorafits.nirda import NIRDALevel1HDUList

# One minute cadence, 10 cadences, 10 minute file
JD = 2460000 + np.arange(10) / (24 * 60)


def test_time_within_fraction_all_and_none():
    distance = np.full(10, 5.0)
    assert np.isclose(time_within_fraction(JD, distance, 10, 10), 1)
    assert np.isclose(time_within_fraction(JD, distance, 2, 10), 0)


def test_time_within_fraction_partial_and_nan():
    distance = np.full(10, 5.0)
    distance[:3] = 25
    distance[3:5] = np.nan
    assert np.isclose(time_within_fraction(JD, distance, 10, 10), 0.5)
    assert np.isclose(time_within_fraction(JD, distance, 30, 10), 0.8)


def test_time_within_fraction_gap():
    # Drop cadences 4-7; the gap counts against the fraction.
    keep = np.r_[0:4, 8:10]
    distance = np.full(len(keep), 5.0)
    assert np.isclose(time_within_fraction(JD[keep], distance, 10, 10), 0.6)


def test_time_within_fraction_edge_cases():
    assert time_within_fraction(JD[:0], np.array([]), 10, 10) == 0
    assert time_within_fraction(JD, np.full(10, 5.0), 10, 0) == 0
    assert np.isclose(time_within_fraction(JD[:1], np.array([5.0]), 10, 1), 1)


def make_nirda_hdulist(target_pixels):
    """Synthetic NIRDA file with 100 one-minute cadences on target and a
    target aperture of `target_pixels` pixels of 10 ct/s above the sky."""
    nframes, shape = 100, (6, 8)
    aperture = np.zeros(shape, np.int16)
    aperture.flat[:target_pixels] = 2
    science = np.full((nframes, *shape), 5.0) + 10.0 * (aperture == 2)
    vectors = fits.BinTableHDU.from_columns(
        [
            fits.Column(name, "D", array=np.full(nframes, value))
            for name, value in [
                ("target_sep", 0.0),
                ("avg_ra", 10.0),
                ("avg_dec", 20.0),
                ("err_ra", 1 / 3600),
                ("err_dec", 0.0),
            ]
        ],
        name="VECTORS",
    )
    header = fits.Header([("PFCLASS", "NIRDALevel1HDUList")])
    return NIRDALevel1HDUList(
        fits.HDUList(
            [
                fits.PrimaryHDU(header=header),
                fits.ImageHDU(science, name="SCIENCE"),
                fits.ImageHDU(
                    np.full((nframes, shape[0]), 5.0), name="BACKGROUND"
                ),
                fits.ImageHDU(np.arange(nframes), name="INTEGRATION"),
                fits.ImageHDU(
                    JD[0] + np.arange(nframes) / (24 * 60), name="TIME"
                ),
                fits.ImageHDU(
                    np.full(nframes, 4.74),
                    header=fits.Header([("UNIT", "second")]),
                    name="EXPTIME",
                ),
                vectors,
                fits.ImageHDU(aperture, name="APERTURE"),
                fits.ImageHDU(np.zeros(shape, np.int16), name="QUALITY"),
            ]
        ),
        validate=False,
    )


def test_snr_inspector_with_target_aperture():
    cards = {
        c.keyword: c.value
        for c in SNRInspector(make_nirda_hdulist(4)).calculate()
    }
    assert np.isclose(cards["LC_MED"], 40)
    assert np.isclose(cards["MAX_PIX"], 10)
    assert np.isclose(cards["PRECSN"], 1)


def test_snr_inspector_with_empty_target_aperture():
    cards = {
        c.keyword: c.value
        for c in SNRInspector(make_nirda_hdulist(0)).calculate()
    }
    assert cards["LC_MED"] == 0
    assert cards["MAX_PIX"] == 0
    # Pointing precision does not depend on the aperture.
    assert np.isclose(cards["PRECSN"], 1)
