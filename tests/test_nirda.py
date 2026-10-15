# Third-party
import numpy as np
import pytest
from astropy.io import fits

# First-party/Local
from pandorafits.nirda import NIRDALevel1HDUList, get_nirda_acqd_ints


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


def nirda_header(integrations, frmsacq, groups_averaged=1):
    return fits.Header(
        [
            ("INTEGRTS", integrations),
            ("GRPS", 2),
            ("READS", 4),
            ("GRPSAVGD", groups_averaged),
            ("FRMSEXP", integrations * 8),
            ("FRMSACQ", frmsacq),
        ]
    )


@pytest.mark.parametrize(
    "integrations, frmsacq, nframes, groups_averaged, expected",
    [
        # Complete file.
        (10, 80, 20, 1, 10),
        (10, 80, 80, 0, 10),
        # Truncated after 5 complete integrations (40 reads); the old code
        # kept 3.
        (10, 40, 10, 1, 5),
        (10, 43, 11, 1, 5),
        (10, 43, 43, 0, 5),
        # FRMSACQ claims 246 integrations but the data hold 491 averaged
        # groups (245 integrations), as in real files.
        (349, 1971, 491, 1, 245),
        # Without the data length only FRMSACQ is used.
        (349, 1971, None, 1, 246),
    ],
)
def test_nirda_complete_integrations(
    integrations, frmsacq, nframes, groups_averaged, expected
):
    header = nirda_header(integrations, frmsacq, groups_averaged)
    assert get_nirda_acqd_ints(header, nframes) == expected
