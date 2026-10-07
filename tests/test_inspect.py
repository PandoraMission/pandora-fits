# Third-party
import numpy as np

# First-party/Local
from pandorafits.inspect import time_within_fraction

# One minute cadence, 10 cadences, 10 minute file
JD = 2460000 + np.arange(10) / (24 * 60)


def test_time_within_fraction_all_and_none():
    distance = np.full(10, 5.0)
    assert np.isclose(time_within_fraction(JD, distance, 10, 10), 1)
    assert np.isclose(time_within_fraction(JD, distance, 2, 10), 0)


def test_time_within_fraction_partial_and_nan():
    distance = np.full(10, 5.0)
    distance[:3] = 25
    distance[3:5] = np.nan
    assert np.isclose(time_within_fraction(JD, distance, 10, 10), 0.5)
    assert np.isclose(time_within_fraction(JD, distance, 30, 10), 0.8)


def test_time_within_fraction_gap():
    # Drop cadences 4-7; the gap counts against the fraction.
    keep = np.r_[0:4, 8:10]
    distance = np.full(len(keep), 5.0)
    assert np.isclose(time_within_fraction(JD[keep], distance, 10, 10), 0.6)


def test_time_within_fraction_edge_cases():
    assert time_within_fraction(JD[:0], np.array([]), 10, 10) == 0
    assert time_within_fraction(JD, np.full(10, 5.0), 10, 0) == 0
    assert np.isclose(time_within_fraction(JD[:1], np.array([5.0]), 10, 1), 1)
