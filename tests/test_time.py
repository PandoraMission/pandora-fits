# Third-party
import numpy as np
from astropy.io import fits

# First-party/Local
from pandorafits.nirda import NIRDALevel0HDUList
from pandorafits.utils import convert_time
from pandorafits.visda import VISDALevel0HDUList

FRAME_SECONDS = 0.237


def make_nirda_level0(integrations=2):
    # The flight readout: 253 resets, 1 drop, then two groups of 4 averaged
    # reads separated by 16 drops; later integrations have 1 reset.
    header = fits.Header(
        [
            ("CORSTIME", 842115335),
            ("FINETIME", 0),
            ("FRMTIME", FRAME_SECONDS * 1000),
            ("EXPOSRES", 1),
            ("INTEGRTS", integrations),
            ("GRPS", 2),
            ("READS", 4),
            ("GRPSAVGD", 1),
            ("RESETS1", 253),
            ("RESETS2", 1),
            ("DROPS1", 1),
            ("DROPS2", 16),
            ("DROPS3", 0),
            ("FRMSEXP", integrations * 8),
            ("FRMSACQ", integrations * 8),
        ]
    )
    data = np.zeros((integrations * 2, 3, 4), dtype=np.int16)
    return NIRDALevel0HDUList(
        fits.HDUList(
            [
                fits.PrimaryHDU(header=header),
                fits.ImageHDU(data, name="SCIENCE"),
            ]
        ),
        validate=False,
    )


def frames_after_start(hdulist, time_jd):
    start = convert_time(hdulist[0].header["CORSTIME"], 0)
    return (time_jd - start.jd) * 86400 / FRAME_SECONDS


def test_nirda_time_is_mid_exposure():
    hdulist = make_nirda_level0().fix()
    # Groups are read in frames 254-257 and 274-277 (first integration) and
    # 280-283 and 300-303 (second); each group is stamped at the mean of its
    # reads' midpoints.
    groups = frames_after_start(hdulist, hdulist["TIME"].data[:, :, 0])
    assert np.allclose(groups, [[256, 276], [282, 302]], atol=1e-3)

    hdulist = hdulist.difference_sample()
    assert hdulist["TIME"].header["TIMEPIXR"] == 0.5
    differences = frames_after_start(hdulist, hdulist["TIME"].data)
    assert np.allclose(differences, [266, 292], atol=1e-3)
    assert np.allclose(hdulist["EXPTIME"].data, 20 * FRAME_SECONDS)


def make_visda_level0():
    header = fits.Header(
        [
            ("INSTRMNT", "VISDA"),
            ("CORSTIME", 842115335),
            ("FINETIME", 0),
            ("EXPTIME", 200000),  # microseconds per read
            ("FRMPCOAD", 50),
            ("NUMPCOAD", 3),
        ]
    )
    return VISDALevel0HDUList(
        fits.HDUList(
            [
                fits.PrimaryHDU(header=header),
                fits.ImageHDU(np.zeros((3, 4, 4)), name="SCIENCE"),
            ]
        ),
        validate=False,
    )


def test_visda_time_is_mid_exposure():
    hdulist = make_visda_level0().fix()
    start = convert_time(842115335, 0)
    seconds = (hdulist["TIME"].data - start.jd) * 86400
    # Co-adds of 50 reads of 0.2 s, stamped at their middle.
    assert np.allclose(seconds, [5, 15, 25], atol=1e-3)
    assert hdulist["TIME"].header["TIMEPIXR"] == 0.5


def test_exposure_windows_follow_timepixr():
    hdulist = make_visda_level0().fix()
    time_jd = hdulist["TIME"].data
    ten_seconds = 10 / 86400

    start, end = hdulist.exposure_windows
    assert np.allclose(start.jd, time_jd - ten_seconds / 2, atol=1e-9)
    assert np.allclose(end.jd, time_jd + ten_seconds / 2, atol=1e-9)

    # Files from before TIMEPIXR stamped the start of each exposure.
    for legacy in [None, ""]:
        if legacy is None:
            del hdulist["TIME"].header["TIMEPIXR"]
        else:
            hdulist["TIME"].header["TIMEPIXR"] = legacy
        start, end = hdulist.exposure_windows
        assert np.allclose(start.jd, time_jd, atol=1e-9)
        assert np.allclose(end.jd, time_jd + ten_seconds, atol=1e-9)
