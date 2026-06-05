# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from . import FORMATSDIR, NIRDAReference, logger
from .fits import PandoraHDUList
from .io import register_hdulist
from .report import ReportMixins
from .scene import get_NIRDA_scene
from .utils import convert_time

__all__ = [
    "NIRDALevel0HDUList",
    "NIRDALevel1HDUList",
    "NIRDALevel2HDUList",
]


def get_nirda_frame_indexes_tot(hdr):
    totframes = 0
    times = []
    for e in range(hdr["EXPOSRES"]):
        for i in range(hdr["INTEGRTS"]):
            if i == 0:
                totframes += hdr["RESETS1"]
            else:
                totframes += hdr["RESETS2"]
            totframes += hdr["DROPS1"]
            for g in range(hdr["GRPS"]):
                if g != 0:
                    totframes += hdr["DROPS2"]
                times.extend(totframes + np.arange(hdr["READS"]))
                totframes += hdr["READS"]
                if g == hdr["GRPS"] - 1:
                    totframes += hdr["DROPS3"]
    return np.asarray(times)


def get_nirda_frame_times(hdr):
    indexes = get_nirda_frame_indexes_tot(hdr)
    return convert_time(hdr["CORSTIME"], hdr["FINETIME"]) + (
        indexes * hdr["FRMTIME"] * u.millisecond
    )


def get_nirda_read_frame_indexes(hdr):
    totframes = 0
    times = []
    for e in range(hdr["EXPOSRES"]):
        for i in range(hdr["INTEGRTS"]):
            for g in range(hdr["GRPS"]):
                times.extend(totframes + np.arange(hdr["READS"]))
                totframes += hdr["READS"]
    return np.asarray(times)


def get_nirda_acqd_ints(hdr0):
    if hdr0["FRMSACQ"] != hdr0["FRMSEXP"]:
        idxs = get_nirda_read_frame_indexes(hdr0)
        idxs = idxs.reshape((hdr0["INTEGRTS"], hdr0["GRPS"], hdr0["READS"]))
        acqd_reads = idxs < (hdr0["FRMSACQ"] - 1)
        acqd_ints = np.where(acqd_reads.all(axis=(1, 2)))[0][-1]
    else:
        acqd_ints = hdr0["INTEGRTS"]
    return acqd_ints


def get_nirda_exposure_times(hdr):
    times = []
    for e in range(hdr["EXPOSRES"]):
        for i in range(hdr["INTEGRTS"]):
            totframes = 0
            for g in range(hdr["GRPS"]):
                if g != 0:
                    totframes += hdr["DROPS2"]
                times.extend(totframes + np.arange(1, hdr["READS"] + 1))
                totframes += hdr["READS"]
    return np.asarray(times)


@register_hdulist(
    lambda h: (
        h
        and (
            (h[0].header.get("TELESCOP") == "NASA Pandora")
            & (h[0].header.get("INSTRMNT") == "NIRDA")
            & ("PFCLASS" not in h[0].header)
        )
    )
)
class NIRDALevel0HDUList(ReportMixins, PandoraHDUList):
    filename = FORMATSDIR + "nirda/level0_nirda.xlsx"
    reference = NIRDAReference
    level = 0
    instrument = "NIRDA"

    def __to_l1__(self):
        return NIRDALevel1HDUList(self)

    @property
    def hot_pixels(self):
        # Count pixels per frame exceeding median + 5 x sigma_MAD across the
        # full frame.  MAD x 1.4826 gives a Gaussian-equivalent sigma robust to
        # the outliers being detected (hot pixels barely move the median).
        science = self["science"].data.astype(float)  # (nframes, roi_y, roi_x)
        pixels = science.reshape(self.nframes, -1)
        med = np.median(pixels, axis=1, keepdims=True)
        sigma_mad = 1.4826 * np.median(np.abs(pixels - med), axis=1, keepdims=True)
        counts = (pixels > med + 5.0 * sigma_mad).sum(axis=1)
        return self.time, counts

    @property
    def dead_pixels(self):
        science = self["science"].data  # (nframes, roi_y, roi_x)
        counts = (science == 0).sum(axis=(1, 2))
        return self.time, counts

    def noise_psd(self, roi_xdelta_right=5, roi_xdelta_left=5):
        """Return the averaged one-sided PSD of background pixel time series.

        Background pixels are taken from the top and bottom ``roi_xdelta_*``
        columns of the spatial axis (roi_x), where the stellar PSF has fallen
        off.  Combining both edges hedges against detector column variations
        and stray sources landing in one region.

        Returns
        -------
        freq : ndarray
            Frequency array in Hz (length nfreqs, DC=0 excluded).
        psd : ndarray
            Mean one-sided PSD in counts^2 / Hz (length nfreqs).
        """
        science = self["science"].data.astype(float)  # (nframes, roi_y, roi_x)
        bg = np.concatenate(
            [science[:, :, :roi_xdelta_left], science[:, :, -roi_xdelta_right:]],
            axis=2,
        )  # (nframes, roi_y, n_bg_cols)

        # Flatten roi dimensions; each column is a pixel time series of length nframes
        pixels = bg.reshape(self.nframes, -1)  # (nframes, n_bg_pixels)
        # Remove per-pixel DC so the PSD captures AC noise only
        pixels -= pixels.mean(axis=0, keepdims=True)

        dt = self.frame_time.to(u.second).value
        n = self.nframes
        fs = 1.0 / dt

        fft_vals = np.fft.rfft(pixels, axis=0)  # (nfreqs, n_bg_pixels)
        # One-sided PSD normalisation: P(f) = 2|X(f)|^2 / (N * fs)
        # Factor of 2 folds negative frequencies onto positive side.
        # DC and Nyquist bins are not doubled (they have no negative counterpart).
        psd = (np.abs(fft_vals) ** 2) / (n * fs)
        psd[1:-1] *= 2

        freq = np.fft.rfftfreq(n, d=dt)
        # Return without the DC bin (freq=0 → undefined on log scale)
        return freq[1:], psd[1:].mean(axis=1)

    def plot_noise_psd(self, ax=None, roi_xdelta_right=5, roi_xdelta_left=5, **kwargs):
        called_from_report = ax is not None
        if ax is None:
            _, ax = plt.subplots()
        freq, psd = self.noise_psd(
            roi_xdelta_right=roi_xdelta_right, roi_xdelta_left=roi_xdelta_left
        )
        ax.loglog(freq, psd, **kwargs)
        # Overlay a 1/f reference line anchored to the lowest frequency bin
        ref = psd[0] * (freq[0] / freq)
        ax.loglog(freq, ref, ls="--", color="gray", lw=1, label="1/f ref")
        ax.legend()
        ax.set(xlabel="Frequency [Hz]", ylabel="PSD [counts² / Hz]")
        if not called_from_report:
            ax.set(title=f"{self[0].header['targ_id']} {self.start_time.isot}")
        return ax

    def plot_bad_pixels(self, ax=None, **kwargs):
        called_from_report = ax is not None
        if ax is None:
            _, ax = plt.subplots()
        t, hot = self.hot_pixels
        _, dead = self.dead_pixels
        t_min = (t.jd - self.start_time.jd) * 24 * 60
        ax.plot(t_min, hot, label="Hot", **kwargs)
        ax.plot(t_min, dead, label="Dead", **kwargs)
        ax.legend()
        ax.set(
            yscale='log',
            xlabel="Time from Start [min]",
            ylabel="Pixel Count")
        if not called_from_report:
            ax.set(title=f"{self[0].header['targ_id']} {self.start_time.isot}")
        return ax

    def describe(self):
        keys = [
            "TARG_ID",
            "TARG_RA",
            "TARG_DEC",
            "INTEGRTS",
            "GRPS",
            "READS",
            "EXPOSRES",
            "ROISIZEX",
            "ROISIZEY",
        ]
        hdr = self[0].header
        rows = []
        for key in keys:
            try:
                value = hdr[key]
                comment = hdr.comments[key]
            except (KeyError, IndexError):
                value, comment = "N/A", ""
            if key in ("TARG_RA", "TARG_DEC"):
                try:
                    value = round(float(value), 4)
                except (TypeError, ValueError):
                    pass
            rows.append([key, value, comment])
        _, hot = self.hot_pixels
        _, dead = self.dead_pixels
        rows.append(["HOT_MED", int(np.median(hot)), "Median hot pixels per frame"])
        rows.append(["DEAD_MED", int(np.median(dead)), "Median dead pixels per frame"])
        return pd.DataFrame(
            rows, columns=["Key", "Value", "Comment"]
        ).set_index("Key")

    def get_report_materials(self):
        return {
            "title": self[0].header.get("TARG_ID", "NIRDA"),
            "subtitle": self.start_time.isot,
            "tables": self.describe(),
            "star_field": lambda ax=None: self.plot_data(ax=ax),
            "astrometry": lambda ax=None: self.plot_noise_psd(ax=ax),
            "bad_pixels": lambda ax=None: self.plot_bad_pixels(ax=ax),
        }

    def plot_data(self, ax=None, **kwargs):
        called_from_report = ax is not None
        if ax is None:
            _, ax = plt.subplots()
        
        use_log = kwargs.pop("log", True)
        if use_log:
            d = np.median(np.log(self["science"].data), axis=0)
            max_add = 0.3
        else:
            d = np.median(self["science"].data, axis=0)
            max_add = 100.0
        k = d != 0
        vmin = kwargs.pop("vmin", np.nanpercentile(d[k], 1.0))
        vmax = kwargs.pop("vmax", np.nanpercentile(d[k], 1.0) + max_add) 
        im = ax.pcolormesh(
            self.column, self.row, d, vmin=vmin, vmax=vmax, **kwargs
        )
        ax.set(
            aspect="equal",
            xlabel="ROI Column",
            ylabel="ROI Row",
        )
        if not called_from_report:
            ax.set(title=f"{self[0].header['targ_id']} {self.start_time.isot}")
        plt.colorbar(im, ax=ax)
        # ax.margins(0)
        return ax


@register_hdulist(
    lambda h: (
        h
        and (
            (h[0].header.get("TELESCOP") == "NASA Pandora")
            & (h[0].header.get("INSTRMNT") == "NIRDA")
            & (h[0].header.get("PFCLASS") == "NIRDALevel1HDUList")
        )
    )
)
class NIRDALevel1HDUList(NIRDALevel0HDUList):
    filename = FORMATSDIR + "nirda/level1_nirda.xlsx"
    level = 1

    # def get_scene(self):
    #     hdr = self[0].header
    #     prf = pa.DispersedPRF.from_reference()
    #     prf.imcorner = (hdr["ROISTRTY"], hdr["ROISTRTX"])
    #     prf.imshape = (hdr["ROISIZEY"], hdr["ROISIZEX"])
    #     scene = pa.DispersedSkyScene(prf, self.wcs, self.start_time)
    #     return scene

    def get_scene(self):
        hdr = self[0].header
        imcorner = (hdr["ROISTRTY"], hdr["ROISTRTX"])
        imshape = (hdr["ROISIZEY"], hdr["ROISIZEX"])
        return get_NIRDA_scene(
            time_jd=self.sequence_start_time.jd,
            ra=hdr["TARG_RA"],
            dec=hdr["TARG_DEC"],
            roll=hdr["TARG_RLL"],
            imcorner=imcorner,
            imshape=imshape,
        )

    def _append_scene_extensions(self):
        hdr = self[0].header
        scene = self.get_scene()
        self.append(scene.get_catalog_hdu())
        self.append(scene.get_prf_hdu())
        self.append(scene.get_model_hdu())
        self.append(
            scene.get_aperture_hdu(
                SkyCoord(hdr["targ_ra"], hdr["targ_dec"], unit="deg"),
                relative_threshold=0.0001,
                absolute_threshold=30,
            )
        )
        self[0].header["GAIA_ID"] = self["APERTURE"].header["GAIA_ID"]
        logger.info("Appended scene extensions")

    def __to_l2__(self):
        return NIRDALevel2HDUList(self)


@register_hdulist(
    lambda h: (
        h
        and (
            (h[0].header.get("TELESCOP") == "NASA Pandora")
            & (h[0].header.get("INSTRMNT") == "NIRDA")
            & (h[0].header.get("PFCLASS") == "NIRDALevel2HDUList")
        )
    )
)
class NIRDALevel2HDUList(NIRDALevel1HDUList):
    filename = FORMATSDIR + "nirda/level2_nirda.xlsx"
    level = 2
