"""Module to deal with SC roll"""

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, get_sun
from astropy.time import Time

__all__ = ["get_roll"]


def get_roll(time: Time, target: SkyCoord):
    target_vec = target.cartesian.get_xyz().value
    if target_vec.ndim == 1:
        target_vec = target_vec[None, :]
    else:
        target_vec = target_vec.T

    sun = get_sun(time)
    sun_vec = sun.cartesian.get_xyz().value / sun.cartesian.norm().value
    if sun_vec.ndim == 1:
        sun_vec = sun_vec[None, :]
    else:
        sun_vec = sun_vec.T

    # This is the direction the solar panels must point
    yB = np.cross(sun_vec, target_vec)

    # breaks in some cases come back to this>
    yB = yB / np.linalg.norm(yB, axis=1)[:, None]

    # The other dimension is the cross of these two
    xB = np.cross(yB, target_vec)
    xB /= np.linalg.norm(yB, axis=1)[:, None]

    # Celestial north in ECI (equatorial frame) is +Z_ECI
    north = np.array([0.0, 0.0, 1.0])
    north_proj = north[None, :] - np.dot(target_vec, north)[:, None] * target_vec

    # reference axes for roll angle calculation

    # The x_ref is the normalized north projectiong
    x_ref = north_proj / np.linalg.norm(north_proj, axis=1)[:, None]

    # The y_ref is orthogonal to the target and x_ref
    y_ref = np.cross(target_vec, x_ref)
    y_ref = y_ref / np.linalg.norm(y_ref, axis=1)[:, None]

    # Compute roll angle x_ref -> xB around zB
    # To do this we calculate the position of xB in the x_ref, y_ref plane
    xB_a = np.sum(x_ref * xB, axis=1)
    xB_b = np.sum(y_ref * xB, axis=1)
    roll = np.rad2deg(np.arctan2(xB_b, xB_a))  # degrees
    return roll * u.deg
