"""Test we can reshape arrays back and forth"""

# Third-party
import numpy as np

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
