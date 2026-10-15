# Standard library
import sqlite3

# Third-party
import numpy as np
import pytest

# First-party/Local
from pandorafits.utils import (
    CircularMeanAggregate,
    circular_mean_deg,
    circular_median_deg,
    wrap_deg,
)

# Roll samples straddling 0/360 deg; the arithmetic mean is 180.
STRADDLING = np.array([359.9, 0.1] * 50)


def angle_difference(a, b):
    return abs(wrap_deg(a - b))


def test_wrap_deg():
    assert np.allclose(
        wrap_deg([0, 179, 180, 359, -1, 721]), [0, 179, -180, -1, -1, 1]
    )


@pytest.mark.parametrize("statistic", [circular_mean_deg, circular_median_deg])
def test_circular_statistics_straddling_zero(statistic):
    assert np.isclose(np.mean(STRADDLING), 180)
    assert angle_difference(statistic(STRADDLING), 0) < 1e-9
    assert angle_difference(statistic([350, 10, np.nan]), 0) < 1e-9
    assert np.isclose(statistic([90, 90]), 90)
    assert 0 <= statistic([-10, -10]) < 360
    assert np.isnan(statistic([np.nan]))


def test_circular_median_is_robust_to_glitches():
    # A single-sample attitude glitch moves the mean but not the median.
    rolls = np.append(np.full(20, 1.0), 213.0)
    assert angle_difference(circular_mean_deg(rolls), 1) > 1
    assert np.isclose(circular_median_deg(rolls), 1)


def test_circular_mean_sql_aggregate():
    connection = sqlite3.connect(":memory:")
    connection.create_aggregate("CIRCULAR_AVG", 1, CircularMeanAggregate)
    connection.execute("CREATE TABLE attitude (roll FLOAT)")
    connection.executemany(
        "INSERT INTO attitude VALUES (?)",
        [(float(roll),) for roll in STRADDLING] + [(None,)],
    )
    circular, arithmetic = connection.execute(
        "SELECT CIRCULAR_AVG(roll), AVG(roll) FROM attitude"
    ).fetchone()
    assert angle_difference(circular, 0) < 1e-9
    assert np.isclose(arithmetic, 180)
    empty = connection.execute(
        "SELECT CIRCULAR_AVG(roll) FROM attitude WHERE roll IS NULL"
    ).fetchone()[0]
    assert empty is None
