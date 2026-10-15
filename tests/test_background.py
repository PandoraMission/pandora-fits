# Third-party
import numpy as np
from astropy.io import fits

# First-party/Local
from pandorafits.nirda import NIRDALevel1HDUList

NFRAMES, NROWS, NCOLS = 3, 20, 30
TRACE_COLUMNS = slice(12, 18)
HOT_PIXEL = (10, 5)


def make_nirda_hdulist():
    """Synthetic NIRDA file: noisy sky, bright subarray edge columns and
    edge rows, a target trace, and one unflagged hot pixel."""
    image = np.random.default_rng(0).normal(75, 1, (NROWS, NCOLS))
    image[:, [0, -1]] = 300
    image[0] += 165
    image[-1] += 235
    image[:, TRACE_COLUMNS] += 5000
    image[HOT_PIXEL] = 1e4
    science = np.repeat(image[None], NFRAMES, axis=0)
    science += np.arange(NFRAMES)[:, None, None]

    aperture = np.zeros((NROWS, NCOLS), np.int16)
    aperture[:, TRACE_COLUMNS] = 2
    quality = np.zeros((NROWS, NCOLS), np.int16)
    return NIRDALevel1HDUList(
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(science, name="SCIENCE"),
                fits.ImageHDU(aperture, name="APERTURE"),
                fits.ImageHDU(quality, name="QUALITY"),
            ]
        ),
        validate=False,
    )


def test_nirda_background_excludes_edge_columns_and_outliers():
    hdulist = make_nirda_hdulist()
    background = hdulist.get_bkg()
    assert background.shape == (NFRAMES, NROWS)

    # Expected: median of the sky columns, without the two edge columns on
    # each side, the trace, and the hot pixel.
    sky = np.ones(NCOLS, bool)
    sky[:2] = sky[-2:] = False
    sky[TRACE_COLUMNS] = False
    data = hdulist["SCIENCE"].data
    for row in range(NROWS):
        columns = sky.copy()
        if row == HOT_PIXEL[0]:
            columns[HOT_PIXEL[1]] = False
        expected = np.median(data[:, row, columns], axis=1)
        assert np.allclose(background[:, row], expected)


def test_nirda_background_keeps_edge_rows():
    background = make_nirda_hdulist().get_bkg()
    assert np.isfinite(background).all()
    # The bright edge rows get their own (higher) background.
    assert np.allclose(background[:, 0] - background[:, 1], 165, atol=1)
    assert np.allclose(background[:, -1] - background[:, 1], 235, atol=1)
