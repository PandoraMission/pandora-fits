"""Functions to convert VisSci shapes to other formats"""

import numpy as np

__all__ = ["list_to_panels", "panels_to_cube", "cube_to_list", "panels_to_list"]


def list_to_panels(data, border=True):
    dim = data.ndim
    if dim == 3:
        data = data[None, :, :, :]

    num_frames, num_stars = data.shape[:2]
    shape = data.shape[2]
    num_sub_frames = int(np.ceil(np.sqrt(num_stars)))
    ex = int(border)
    dCube = np.zeros(
        (
            num_frames,
            num_sub_frames * (shape + ex) + ex,
            num_sub_frames * (shape + ex) + ex,
        ),
        dtype=data.dtype,
    )

    star = 0
    while star < num_stars:
        for i in range(num_sub_frames):
            yStrt = (num_sub_frames - (i + 1)) * (shape + ex) + ex
            for j in range(num_sub_frames):
                xStrt = j * (shape + ex) + ex
                dCube[:, yStrt : yStrt + shape, xStrt : xStrt + shape] = data[
                    :, star, :, :
                ]
                star += 1
                if star >= num_stars:
                    break
            if star >= num_stars:
                break
    if dim == 3:
        return dCube[0]
    return dCube


def panels_to_cube(data, nROI, ROI_size):
    numSubFrms = int(np.ceil(np.sqrt(nROI)))
    dims = (numSubFrms * ROI_size[0], numSubFrms * ROI_size[1])
    if data.shape[1] == dims[0]:
        ex = 0
    elif data.shape[1] == (dims[0] + numSubFrms + 1):
        ex = 1
    else:
        raise ValueError("Can not parse the padding dimensions.")
    d = data[:, ex:, ex:]
    shape = d.shape[1:]
    width = ROI_size[0]
    nims = int(shape[1] / (width + ex))
    d1 = np.asarray(np.array_split(d, nims, axis=2))
    nims = int(shape[0] / (width + ex))
    d2 = np.asarray(np.array_split(d1, nims, axis=2))[:, :, :, :-ex, :-ex].transpose(
        [2, 0, 1, 3, 4]
    )
    return d2


def cube_to_list(cube):
    d = cube[:, ::-1]
    starlist = []
    numSubFrms = cube.shape[1]
    for idx in range(numSubFrms):
        for jdx in range(numSubFrms):
            if (idx * numSubFrms + jdx) == numSubFrms**2:
                break
            starlist.append(d[:, idx, jdx])
    return np.asarray(starlist, dtype=starlist[0].dtype).transpose([1, 0, 2, 3])


def panels_to_list(data, nROI, ROI_size):
    return cube_to_list(panels_to_cube(data, nROI, ROI_size))
