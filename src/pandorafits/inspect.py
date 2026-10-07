"""Submodule for scoring files for SOC

This file should contain hdulist contained functions that run on Level 1 files only, as the last step in processing to level 2.
If you need to use significant compute to calculate these metrics, talk with Christina first before adding them..!

Here "score" is used to mean any single value metric that describes something about the file. Your function might return multiple scores.

"""

# Standard library
from abc import ABC, abstractmethod

# Third-party
import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.table import Table

__all__ = [
    "BackgroundInspector",
    "PositionOnTargetInspector",
    "TimeOnTargetInspector",
    "DetectorTemperatureInspector",
    "SNRInspector",
    "time_within_fraction",
]


def time_within_fraction(jd, distance, threshold, total_minutes):
    """Fraction of an observation that `distance` is below `threshold`.

    Each cadence is weighted by the time until the next cadence, capped at
    the median cadence spacing so that gaps from dropped frames count against
    the fraction. The last cadence is given the median spacing. NaN distances
    (e.g. no valid pointing) never count as within `threshold`.

    Parameters
    ----------
    jd : np.ndarray
        Cadence times [JD].
    distance : np.ndarray
        Distance at each cadence, in the same units as `threshold`.
    threshold : float
        Maximum distance to count a cadence as within.
    total_minutes : float
        Duration of the observation [minutes].

    Returns
    -------
    float
        Fraction between 0 and 1.
    """
    if (len(jd) == 0) or (total_minutes <= 0):
        return 0.0
    cadence_minutes = np.diff(jd) * (24 * 60)
    typical_minutes = (
        np.median(cadence_minutes) if len(cadence_minutes) else total_minutes
    )
    durations = np.append(
        np.minimum(cadence_minutes, typical_minutes), typical_minutes
    )
    within_minutes = durations[distance < threshold].sum()
    # The last cadence's assumed duration can overrun the file end time.
    return float(min(within_minutes / total_minutes, 1.0))


class Inspector(ABC):
    """This Abstract Base Class enforces the rules for designing new types of scores.
    You can subclass it to write your own, see below."""

    def __init__(self, hdulist):
        if not isinstance(hdulist, fits.HDUList):
            raise TypeError("You can only score a FITS file")
        if "PFCLASS" not in hdulist[0].header:
            raise TypeError("You can only score a Pandora FITS file")
        if ("level1" not in hdulist[0].header["PFCLASS"].lower()) & (
            "level2" not in hdulist[0].header["PFCLASS"].lower()
        ):
            raise TypeError("You can only score a low level Pandora FITS file")

        self.hdulist = hdulist

    def __call__(self) -> None:
        self.score()

    def score(self) -> None:

        cards = self.calculate()

        if not isinstance(cards, list):
            raise TypeError(
                "The calculate() method for this score class must return a list"
            )

        if not all(isinstance(card, fits.Card) for card in cards):
            raise TypeError(
                "The calculate() method for this score class must return a list of fits.Card objects"
            )

        self.hdulist[0].header.extend(cards)

    @abstractmethod
    def calculate(self) -> list[fits.Card]:
        """Calculate and return FITS header cards."""
        raise NotImplementedError


class BackgroundInspector(Inspector):
    """Scores the background in the file.
    Provides the 10th, 50th, and 90th percentile of the background
    """

    def calculate(self):
        bkg10, bkg50, bkg90 = np.nanpercentile(
            self.hdulist["BACKGROUND"].data.ravel(), [10, 50, 90]
        )
        return [
            fits.Card(
                "BKG10",
                bkg10,
                "10th percentile of Background",
            ),
            fits.Card(
                "BKG50",
                bkg50,
                "50th percentile of Background",
            ),
            fits.Card(
                "BKG90",
                bkg90,
                "90th percentile of Background",
            ),
        ]


class PositionOnTargetInspector(Inspector):
    """Scores how on target the file is.

    TCENT10/20/30 are the fraction of the observation that the target was
    within 10/20/30 pixels of the subarray center, using the per-cadence
    target position in the VECTORS extension. Cadences without valid VITL
    pointing count as not centered.
    """

    def calculate(self):
        vectors = self.hdulist["VECTORS"].data
        # VECTORS fills posx/posy with the static WCS position when pointing
        # is missing, so mask those cadences out.
        has_pointing = (
            np.isfinite(vectors["avg_ra"])
            & np.isfinite(vectors["avg_dec"])
            & np.isfinite(vectors["avg_rot"])
        )
        center_distance = np.where(
            has_pointing,
            np.hypot(
                vectors["posx"] - self.hdulist.column.mean(),
                vectors["posy"] - self.hdulist.row.mean(),
            ),
            np.nan,
        )
        file_minutes = (
            (self.hdulist.end_time - self.hdulist.start_time)
            .to(u.minute)
            .value
        )
        centered_cards = [
            fits.Card(
                f"TCENT{threshold}",
                time_within_fraction(
                    vectors["jd"], center_distance, threshold, file_minutes
                ),
                f"Fraction of time target within {threshold} px of center",
            )
            for threshold in [10, 20, 30]
        ]

        apcomp = np.nan_to_num(
            self.hdulist.aperture.sum()
            / (self.hdulist["APERTURE"].data & 2 == 2).sum()
        )
        return [
            fits.Card(
                "APCOMP1",
                100 * apcomp,
                "Aperture Completeness Metric 1",
            ),
            *centered_cards,
            fits.Card(
                "TARGCOMP",
                apcomp > 0.7,
                "Over 70% of target aperture pixels usable",
            ),
        ]


class TimeOnTargetInspector(Inspector):
    """Scores the amount of time the file was taking data."""

    def calculate(self):
        useable = (
            np.isfinite(self.hdulist["VECTORS"].data["avg_ra"])
            & (
                self.hdulist["VECTORS"].data["jd"]
                > (self.hdulist["VECTORS"].data["jd"][0] + (5 / (24 * 60)))
            )
            & self.hdulist["VECTORS"].data["visda_keepout"]
            & self.hdulist["VECTORS"].data["nirda_keepout"]
            & (self.hdulist["VECTORS"].data["target_sep"] < (10 / 3600))
            & (
                np.abs(
                    (
                        self.hdulist["VECTORS"].data["target_sep"]
                        - np.nanmedian(
                            self.hdulist["VECTORS"].data["target_sep"]
                        )
                    )
                )
                < (10 / 3600)
            )
        )

        k = np.isfinite(self.hdulist["VECTORS"].data["avg_ra"]) & np.isfinite(
            self.hdulist["VECTORS"].data["avg_dec"]
        )

        return [
            fits.Card(
                "SRT_DATE", self.hdulist.start_time.isot, "File Start Date"
            ),
            fits.Card("END_DATE", self.hdulist.end_time.isot, "File End Date"),
            fits.Card(
                "FILETIME",
                (self.hdulist.end_time - self.hdulist.start_time)
                .to(u.minute)
                .value,
                "Time the file is observed for in minutes",
            ),
            fits.Card(
                "VITLTIME",
                (
                    self.hdulist["VECTORS"].data["jd"][k][-1]
                    - self.hdulist["VECTORS"].data["jd"][k][0]
                )
                * (24 * 60),
                "Time VITL was on in minutes",
            ),
            fits.Card(
                "ONTARG",
                100
                * (
                    self.hdulist["VECTORS"].data["target_sep"] < (10 / 3600)
                ).sum()
                / self.hdulist["VECTORS"].header["NAXIS2"],
                "Percentage On Target",
            ),
            fits.Card(
                "NKEEPOUT",
                100
                * (
                    self.hdulist["VECTORS"].data["nirda_keepout"].sum()
                    / self.hdulist["VECTORS"].header["NAXIS2"]
                ),
                "Percentage of time NIRDA keepout obeyed",
            ),
            fits.Card(
                "VKEEPOUT",
                100
                * (
                    self.hdulist["VECTORS"].data["visda_keepout"].sum()
                    / self.hdulist["VECTORS"].header["NAXIS2"]
                ),
                "Percentage of time VISDA keepout obeyed",
            ),
            fits.Card(
                "VITLMISS",
                self.hdulist["VECTORS"].data["vitl_missing_frames"].sum()
                / (self.hdulist["VECTORS"].data["vitl_frames_count"]).sum(),
                "N times VITL return missing data",
            ),
            fits.Card(
                "TARGTIME",
                np.median(self.hdulist["VECTORS"].data["exptime"])
                / 60
                * useable.sum()
                > 10,
                "Usable Target Time is over 10 minutes",
            ),
        ]


class DetectorTemperatureInspector(Inspector):
    def calculate(self):
        cards = []
        if "TCLDTIP1" in self.hdulist[0].header:
            cards.append(
                fits.Card(
                    "DETTEMP1",
                    self.hdulist[0].header["TCLDTIP1"] < 140,
                    "Cold Tip 1 temperature less than 140K",
                )
            )
        if "TCLDTIP2" in self.hdulist[0].header:
            cards.append(
                fits.Card(
                    "DETTEMP2",
                    self.hdulist[0].header["TCLDTIP2"] < 140,
                    "Cold Tip 2 temperature less than 140K",
                )
            )
        return cards


class SNRInspector(Inspector):
    def calculate(self):
        df = Table(self.hdulist["VECTORS"].data).to_pandas()
        k = np.abs(
            (
                self.hdulist["VECTORS"].data["target_sep"]
                - np.nanmedian(self.hdulist["VECTORS"].data["target_sep"])
            )
        ) < (10 / 3600)
        k &= self.hdulist["VECTORS"].data["target_sep"] < (10 / 3600)
        k &= np.isfinite(self.hdulist["VECTORS"].data["avg_ra"]) & np.isfinite(
            self.hdulist["VECTORS"].data["avg_dec"]
        )
        if self.hdulist.instrument == "VISDA":
            f = (
                self.hdulist["SCIENCE"].data
                - self.hdulist["BACKGROUND"].data[:, None, None]
            )
            bkg_mad = np.median(
                np.abs(
                    np.diff(f[:, self.hdulist.bkg_aperture], axis=0)
                    / np.sqrt(2)
                )
            )
            a = self.hdulist.aperture
            a &= np.median(f, axis=0) > bkg_mad * 10
            lc = f[:, a].sum(axis=1)
        else:
            f = (
                self.hdulist["SCIENCE"].data
                - self.hdulist["BACKGROUND"].data[:, :, None]
            )

            idx = np.unique(
                self.hdulist["INTEGRATION"].data, return_index=True
            )[1]
            lc = (f * self.hdulist.aperture[None, :, :]).sum(axis=(1, 2))[idx]
            k &= self.hdulist.time.jd > (
                self.hdulist.time.jd[0] + (5 / (24 * 60))
            )

        if not k.any():
            lc_med = 0
            lc_mad = 0
            max_pix_val = 0
            precision = 0

        else:
            lc_med = np.nanmedian(lc[k])
            lc_mad = np.nanmedian(np.abs(np.diff(lc[k]) / np.sqrt(2))) * 1.486
            max_pix_val = np.nanmedian(
                f[k][:, self.hdulist.aperture], axis=0
            ).max()
            precision = (
                np.nanmedian(np.hypot(df.err_ra[k], df.err_dec[k])) * 3600
            )

        cards = [
            fits.Card("lc_med", lc_med, "Median light curve flux value"),
            fits.Card("lc_mad", lc_mad, "MAD light curve flux value"),
            fits.Card(
                "max_pix", max_pix_val, "Maximum pixel flux value in aperture"
            ),
            fits.Card("PRECSN", precision, "Pointing precision [arcsecond]"),
            fits.Card(
                "exptime0", self.hdulist.exptime[0].value, "Exposure time [s]"
            ),
        ]
        return cards


def inspect_file(hdulist):
    """Convenience function applies all of the above Inspectors"""
    BackgroundInspector(hdulist).score()
    TimeOnTargetInspector(hdulist).score()
    PositionOnTargetInspector(hdulist).score()
    DetectorTemperatureInspector(hdulist).score()
    SNRInspector(hdulist).score()
