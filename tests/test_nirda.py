# Third-party
import numpy as np
from astropy.io import fits

# First-party/Local
from pandorafits.nirda import NIRDALevel1HDUList


def test_nirda_aperture_masks_pixels_not_rows():
    aperture_bits = np.zeros((6, 8), np.int16)
    aperture_bits[1:5, 2:6] = 2
    # A contaminating trace across two target pixels and one sky pixel.
    aperture_bits[2, 3:5] |= 4
    aperture_bits[2, 7] = 4
    quality = np.zeros((6, 8), np.int16)
    quality[3, 4] = 1
    quality[0, 0] = 1
    hdulist = NIRDALevel1HDUList(
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(aperture_bits, name="APERTURE"),
                fits.ImageHDU(quality, name="QUALITY"),
            ]
        ),
        validate=False,
    )
    expected = (aperture_bits & 2) == 2
    expected[3, 4] = False
    assert np.array_equal(hdulist.aperture, expected)
    # Every target row is kept, including the contaminated and flagged ones.
    assert np.array_equal(hdulist.aperture.any(axis=1), expected.any(axis=1))
