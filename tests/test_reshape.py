"""Test we can reshape arrays back and forth"""

# Third-party
import numpy as np
import pytest

# First-party/Local
import pandorafits as pf


def test_reshape():
    nROI = 10
    ROI_size = (50, 50)
    ar = np.zeros((nROI, 50, 50))
    panels = pf.reshape.array_to_panels(ar, border=True)
    assert panels.ndim == 2
    assert panels.shape == (205, 205)
    cube = pf.reshape.panels_to_cube(panels, nROI=nROI, ROI_size=ROI_size)
    assert cube.ndim == 4
    assert cube.shape == (4, 4, 50, 50)
    cube = pf.reshape.panels_to_cube(
        panels[None, :, :], nROI=nROI, ROI_size=ROI_size
    )
    assert cube.ndim == 5
    assert cube.shape == (1, 4, 4, 50, 50)
    ar2 = pf.reshape.cube_to_array(cube[0], nROI=nROI)
    assert np.allclose(ar, ar2)
    ar2 = pf.reshape.cube_to_array(cube, nROI=nROI)[0]
    assert np.allclose(ar, ar2)
    ar2 = pf.reshape.panels_to_array(panels, nROI=nROI, ROI_size=ROI_size)
    assert np.allclose(ar, ar2)

    ntime = 23
    ar = np.zeros((ntime, nROI, 50, 50))
    panels = pf.reshape.array_to_panels(ar, border=True)
    assert panels.ndim == 3
    assert panels.shape == (ntime, 205, 205)
    cube = pf.reshape.panels_to_cube(panels, nROI=nROI, ROI_size=ROI_size)
    assert cube.ndim == 5
    assert cube.shape == (ntime, 4, 4, 50, 50)
    ar2 = pf.reshape.cube_to_array(cube, nROI=nROI)
    assert np.allclose(ar, ar2)


@pytest.mark.parametrize("border", [True, False])
def test_reshape_roundtrip_keeps_values(border):
    nROI, ROI_size, ntime = 7, (5, 5), 3
    ar = np.arange(ntime * nROI * 25, dtype=float).reshape(
        ntime, nROI, *ROI_size
    )
    panels = pf.reshape.array_to_panels(ar, border=border)
    ex = int(border)
    assert panels.shape == (ntime, 3 * (5 + ex) + ex, 3 * (5 + ex) + ex)
    ar2 = pf.reshape.panels_to_array(panels, nROI=nROI, ROI_size=ROI_size)
    assert np.array_equal(ar, ar2)
