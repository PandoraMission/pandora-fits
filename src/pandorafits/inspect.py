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

__all__ = [
    "BackgroundInspector",
    "PositionOnTargetInspector",
    "TimeOnTargetInspector",
    "DetectorTemperatureInspector",
]


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
    """Scores how on target the file is"""

    def calculate(self):
        mid = np.asarray(
            np.unravel_index(
                np.argmin(
                    np.hypot(
                        self.hdulist.column[None, :]
                        - self.hdulist[0].header["posx"],
                        self.hdulist.row[:, None]
                        - self.hdulist[0].header["posy"],
                    )
                ),
                (self.hdulist.row.shape[0], self.hdulist.column.shape[0]),
            )
        ).astype(float)
        mid[0] -= self.hdulist.row.shape[0] / 2
        mid[1] -= self.hdulist.column.shape[0] / 2

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
            fits.Card("TARGCENT", np.hypot(*mid) < 8, "Target Centered"),
            fits.Card(
                "TARGCOMP",
                (
                    100
                    * self.hdulist.aperture.sum()
                    / (self.hdulist["APERTURE"].data & 2 == 2).sum()
                )
                > 0.7,
                "Target Complete",
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


def inspect_file(hdulist):
    """Convenience function applies all of the above Inspectors"""
    BackgroundInspector(hdulist).score()
    TimeOnTargetInspector(hdulist).score()
    PositionOnTargetInspector(hdulist).score()
    DetectorTemperatureInspector(hdulist).score()
