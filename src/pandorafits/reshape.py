"""Functions to convert VisSci data shapes to other formats"""

# Third-party
import numpy as np

from .docstrings import add_docstring

__all__ = [
    "array_to_panels",
    "panels_to_cube",
    "cube_to_array",
    "panels_to_array",
]


@add_docstring(parameters=["ROI_array", "border"], returns=["ROI_panels"])
def array_to_panels(ROI_array, border=True):
    """Converts an input `ROI_array` array from an array of ROIs to panels."""
    dim = ROI_array.ndim
    if dim == 3:
        ROI_array = ROI_array[None, :, :, :]

    num_frames, num_stars = ROI_array.shape[:2]
    shape = ROI_array.shape[2]
    num_sub_frames = int(np.ceil(np.sqrt(num_stars)))
    ex = int(border)
    dCube = np.zeros(
        (
            num_frames,
            num_sub_frames * (shape + ex) + ex,
            num_sub_frames * (shape + ex) + ex,
        ),
        dtype=ROI_array.dtype,
    )

    star = 0
    while star < num_stars:
        for i in range(num_sub_frames):
            yStrt = (num_sub_frames - (i + 1)) * (shape + ex) + ex
            for j in range(num_sub_frames):
                xStrt = j * (shape + ex) + ex
                dCube[:, yStrt : yStrt + shape, xStrt : xStrt + shape] = (
                    ROI_array[:, star, :, :]
                )
                star += 1
                if star >= num_stars:
                    break
            if star >= num_stars:
                break
    if dim == 3:
        return dCube[0]
    return dCube


@add_docstring(
    parameters=["ROI_panels", "nROI", "ROI_size"], returns=["ROI_cube"]
)
def panels_to_cube(ROI_panels, nROI, ROI_size):
    """Converts an input `ROI_panels` shaped as panels to a data cube with shape (nROI, nROI, *ROI_size)."""

    dim = ROI_panels.ndim
    if dim == 2:
        ROI_panels = ROI_panels[None, :, :]
    numSubFrms = int(np.ceil(np.sqrt(nROI)))
    dims = (numSubFrms * ROI_size[0], numSubFrms * ROI_size[1])
    if ROI_panels.shape[1] == dims[0]:
        ex = 0
    elif ROI_panels.shape[1] == (dims[0] + numSubFrms + 1):
        ex = 1
    else:
        raise ValueError("Can not parse the padding dimensions.")
    d = ROI_panels[:, ex:, ex:]
    shape = d.shape[1:]
    width = ROI_size[0]
    nims = int(shape[1] / (width + ex))
    d1 = np.asarray(np.array_split(d, nims, axis=2))
    nims = int(shape[0] / (width + ex))
    d2 = np.asarray(np.array_split(d1, nims, axis=2))[
        :, :, :, :-ex, :-ex
    ].transpose([2, 0, 1, 3, 4])
    if dim == 2:
        return d2[0]
    return d2


@add_docstring(
    parameters=["ROI_cube", "nROI", "ROI_size"], returns=["ROI_cube"]
)
def cube_to_array(ROI_cube, nROI):
    """Converts an input `ROI_cube` to an array."""
    dim = ROI_cube.ndim
    if dim == 4:
        ROI_cube = ROI_cube[None, :, :, :, :]
    d = ROI_cube[:, ::-1]  # backwards!
    stararray = []
    numSubFrms = ROI_cube.shape[1]
    for idx in range(numSubFrms):
        for jdx in range(numSubFrms):
            if (idx * numSubFrms + jdx) == numSubFrms**2:
                break
            stararray.append(d[:, idx, jdx])
            if len(stararray) == nROI:
                break
        if len(stararray) == nROI:
            break
    ar = np.asarray(stararray, dtype=stararray[0].dtype).transpose(
        [1, 0, 2, 3]
    )
    if dim == 4:
        return ar[0]
    return ar


@add_docstring(
    parameters=["ROI_panels", "nROI", "ROI_size"], returns=["ROI_array"]
)
def panels_to_array(ROI_panels, nROI, ROI_size):
    """Converts an input `ROI_panels` from panels to an array."""
    return cube_to_array(panels_to_cube(ROI_panels, nROI, ROI_size), nROI)
