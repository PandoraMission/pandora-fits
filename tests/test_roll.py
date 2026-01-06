import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time

from pandorafits.database.roll import get_roll


def test_roll():
    time = Time(2461028.197167684, format="jd")
    coord = SkyCoord(285.67942246, 50.241306, unit="deg")
    roll = get_roll(time, coord.from_name("Kepler-10"))
    assert isinstance(roll[0], u.Quantity)
    assert np.isclose(roll[0].value, -18.27724411)
