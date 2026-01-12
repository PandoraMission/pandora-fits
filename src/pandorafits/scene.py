"""This module helps us cache scenes so we don't waste compute regenerating them between files."""

from functools import lru_cache

import numpy as np
import pandoraaperture as pa
from astropy.time import Time

from . import NIRDAReference, VISDAReference

spatial_prf = pa.SpatialPRF.from_reference()
dispersed_prf = pa.DispersedPRF.from_reference()


@lru_cache(maxsize=4)
def get_VISDAFFI_scene(time_jd, ra, dec, roll, imcorner, imshape):
    """Special LRU cached version of a scene so we can reuse it!"""
    spatial_prf.imcorner = imcorner
    spatial_prf.imshape = imshape
    scene = pa.SkyScene(
        spatial_prf,
        VISDAReference.get_wcs(
            target_ra=ra,
            target_dec=dec,
            theta=roll,
            distortion=True,
            yreflect=True,
        ),
        Time(time_jd, format="jd"),
    )
    return scene


@lru_cache(maxsize=4)
def get_VISDA_scene(time_jd, ra, dec, roll, ROI_size, ROI_corners):
    """Special LRU cached version of a scene so we can reuse it!"""
    if len(ROI_corners) == 1:
        prf = pa.SpatialPRF.from_reference().to_PRF(
            np.asarray(ROI_corners[0]) + np.asarray(ROI_size) / 2
        )
        prf.imcorner = ROI_corners[0]
        prf.imshape = ROI_size
        scene = pa.SkyScene(
            prf,
            VISDAReference.get_wcs(
                target_ra=ra,
                target_dec=dec,
                theta=roll,
                distortion=True,
                yreflect=True,
            ),
            Time(time_jd, format="jd"),
        )
    else:
        spatial_prf.imcorner = (0, 0)
        spatial_prf.imshape = (2048, 2048)
        scene = pa.ROISkyScene(
            spatial_prf,
            VISDAReference.get_wcs(
                target_ra=ra,
                target_dec=dec,
                theta=roll,
                distortion=True,
                yreflect=True,
            ),
            Time(time_jd, format="jd"),
            nROIs=len(ROI_corners),
            ROI_size=ROI_size,
            ROI_corners=list(ROI_corners),
        )
    return scene


@lru_cache(maxsize=4)
def get_NIRDA_scene(time_jd, ra, dec, roll, imcorner, imshape):
    """Special LRU cached version of a scene so we can reuse it!"""
    dispersed_prf.imcorner = imcorner
    dispersed_prf.imshape = imshape
    scene = pa.DispersedSkyScene(
        dispersed_prf,
        NIRDAReference.get_wcs(
            target_ra=ra,
            target_dec=dec,
            theta=roll,
            distortion=True,
            yreflect=True,
        ),
        Time(time_jd, format="jd"),
    )
    return scene
