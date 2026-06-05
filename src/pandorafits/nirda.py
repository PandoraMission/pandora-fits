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


def get_integration_times(hdr):
    """Compute the wall-clock duration of the first and subsequent integrations.

    The first integration begins with RESETS1 reset frames; all subsequent
    integrations begin with RESETS2.  The shared base includes drop frames and
    GRPS * READS read frames.  Each detector frame takes pixel_read_time per
    pixel, where pixel count includes the reference pixel border.

    Returns
    -------
    int_1_time : float  Duration of the first integration [s].
    int_n_time : float  Duration of subsequent integrations [s].
    """
    base_frames = (
        hdr["DROPS1"]
        + (hdr["GRPS"] - 1) * hdr["DROPS2"]
        + hdr["DROPS3"]
        + hdr["GRPS"] * hdr["READS"]
    )
    int_1_frames = hdr["RESETS1"] + base_frames
    int_n_frames = hdr["RESETS2"] + base_frames
    pixels = (hdr["ROISIZEX"] + 12) * (hdr["ROISIZEY"] + 2)
    pixel_read_time = 0.00001  # sec / pixel
    frame_time = pixel_read_time * pixels
    return int_1_frames * frame_time, int_n_frames * frame_time


def get_nirda_frames_per_integration(hdr):
    """Number of saved frames per integration reset cycle.

    GRPSAVGD == 0: every read frame is saved → GRPS * READS frames.
    GRPSAVGD != 0: reads averaged within each group → GRPS frames.
    """
    if hdr.get("GRPSAVGD", 1) == 0:
        return hdr["GRPS"] * hdr["READS"]
    return hdr["GRPS"]


def get_nirda_integrations(hdr, science):
    """Sum raw science frames into a per-integration cube.

    Parameters
    ----------
    hdr     : FITS primary header.
    science : ndarray, shape (nframes, roi_y, roi_x).

    Returns
    -------
    ndarray of shape (INTEGRTS, roi_y, roi_x) — one summed frame per
    detector reset cycle.
    """
    fpi = get_nirda_frames_per_integration(hdr)
    n_int = hdr["INTEGRTS"]
    return science.reshape(n_int, fpi, science.shape[1], science.shape[2]).sum(axis=1)


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
    def _frames_per_integration(self):
        return get_nirda_frames_per_integration(self[0].header)

    @property
    def integrations(self):
        """Summed flux cube per integration, shape (INTEGRTS, roi_y, roi_x).

        Delegates to get_nirda_integrations — one summed value per detector
        reset cycle, collapsing the within-ramp read sequence.
        """
        return get_nirda_integrations(
            self[0].header, self["science"].data.astype(float)
        )

    @property
    def integration_times(self):
        """Start time of each integration as an astropy Time array of length INTEGRTS."""
        fpi = self._frames_per_integration
        return self.start_time + np.arange(self[0].header["INTEGRTS"]) * fpi * self.frame_time

    @property
    def hot_pixels(self):
        # Count pixels per integration exceeding median + 5 x sigma_MAD.
        # MAD x 1.4826 gives a Gaussian-equivalent sigma robust to the outliers
        # being detected (hot pixels barely move the median).
        ints = self.integrations  # (n_int, roi_y, roi_x)
        pixels = ints.reshape(len(ints), -1)
        med = np.median(pixels, axis=1, keepdims=True)
        sigma_mad = 1.4826 * np.median(np.abs(pixels - med), axis=1, keepdims=True)
        counts = (pixels > med + 5.0 * sigma_mad).sum(axis=1)
        return self.integration_times, counts

    @property
    def dead_pixels(self):
        # A persistently dead pixel sums to zero across all groups in an integration.
        ints = self.integrations  # (n_int, roi_y, roi_x)
        counts = (ints == 0).sum(axis=(1, 2))
        return self.integration_times, counts

    @property
    def ramp_linearity(self):
        """Measure deviation from a linear ramp within each integration.

        For a well-behaved detector the signal accumulates linearly with group
        number.  A straight line is fitted to the group values per pixel via
        vectorised OLS, and the RMS of the residuals is computed.  The median
        across all pixels gives one scalar per integration.

        Elevated values indicate saturation, persistence, or reset anomalies.

        Returns
        -------
        integration_times : astropy Time array of length INTEGRTS.
        rms_per_int       : ndarray (INTEGRTS,) — median pixel ramp residual RMS.
        """
        science = self["science"].data.astype(float)  # (nframes, roi_y, roi_x)
        fpi = self._frames_per_integration
        n_int = self[0].header["INTEGRTS"]

        # Reshape to (n_int, fpi, roi_y, roi_x) so axis-1 is the within-ramp axis
        ramp = science.reshape(n_int, fpi, science.shape[1], science.shape[2])

        # Vectorised OLS: fit y = a + b*g for g in [0, fpi)
        g = np.arange(fpi, dtype=float)
        g_dev = g - g.mean()
        g_ss = (g_dev ** 2).sum()

        r_mean = ramp.mean(axis=1, keepdims=True)                              # (n_int, 1, h, w)
        slope = (g_dev[None, :, None, None] * (ramp - r_mean)).sum(axis=1) / g_ss  # (n_int, h, w)
        intercept = r_mean[:, 0] - slope * g.mean()                           # (n_int, h, w)

        fitted = intercept[:, None] + slope[:, None] * g[None, :, None, None] # (n_int, fpi, h, w)
        residuals = ramp - fitted

        pixel_rms = np.sqrt((residuals ** 2).mean(axis=1))                    # (n_int, h, w)
        rms_per_int = np.median(pixel_rms.reshape(n_int, -1), axis=1)

        return self.integration_times, rms_per_int

    def noise_psd(self, roi_xdelta_right=5, roi_xdelta_left=5):
        """Return the averaged one-sided PSD of background pixel time series.

        Uses one sample per integration so the frequency axis reflects the
        integration cadence, not the raw frame cadence.

        Returns
        -------
        freq : ndarray  Frequency in Hz (DC bin excluded).
        psd  : ndarray  Mean one-sided PSD in counts^2 / Hz.
        """
        ints = self.integrations  # (n_int, roi_y, roi_x)
        n_int = len(ints)
        bg = np.concatenate(
            [ints[:, :, :roi_xdelta_left], ints[:, :, -roi_xdelta_right:]],
            axis=2,
        )
        pixels = bg.reshape(n_int, -1)
        pixels -= pixels.mean(axis=0, keepdims=True)

        fpi = self._frames_per_integration
        dt = (fpi * self.frame_time).to(u.second).value
        fs = 1.0 / dt

        fft_vals = np.fft.rfft(pixels, axis=0)
        # One-sided PSD: P(f) = 2|X(f)|^2 / (N * fs); DC and Nyquist not doubled.
        psd = (np.abs(fft_vals) ** 2) / (n_int * fs)
        psd[1:-1] *= 2

        freq = np.fft.rfftfreq(n_int, d=dt)
        return freq[1:], psd[1:].mean(axis=1)

    def background_rms(self, roi_xdelta_left=5, roi_xdelta_right=5):
        """Per-integration background RMS after removing each integration's median.

        Returns integration_times and a scalar noise estimate per integration.
        """
        ints = self.integrations  # (n_int, roi_y, roi_x)
        bg = np.concatenate(
            [ints[:, :, :roi_xdelta_left], ints[:, :, -roi_xdelta_right:]],
            axis=2,
        )
        bg -= np.median(bg, axis=(1, 2), keepdims=True)
        rms = np.std(bg, axis=(1, 2))
        return self.integration_times, rms

    def plot_background_rms(self, ax=None, roi_xdelta_left=5, roi_xdelta_right=5, **kwargs):
        called_from_report = ax is not None
        if ax is None:
            _, ax = plt.subplots()
        t, rms = self.background_rms(
            roi_xdelta_left=roi_xdelta_left, roi_xdelta_right=roi_xdelta_right
        )
        t_min = (t.jd - self.start_time.jd) * 24 * 60
        ax.plot(t_min, rms, **kwargs)
        ax.set(
            yscale='linear',
            xlabel="Time from Start [min]",
            ylabel="Background RMS [counts]"
        )
        if not called_from_report:
            ax.set(title=f"{self[0].header['targ_id']} {self.start_time.isot}")
        return ax

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

    def plot_noise_spectrogram(
        self,
        ax=None,
        roi_xdelta_left=5,
        roi_xdelta_right=5,
        nperseg=None,
        **kwargs,
    ):
        """Plot a short-time PSD spectrogram of the background region.

        Splits the background pixel time series into overlapping windows of
        ``nperseg`` integrations (50 % overlap) and computes the PSD for each.
        Averaging across background pixels at each window reduces pixel-level
        scatter so the time evolution of the noise floor is visible.

        Parameters
        ----------
        nperseg : int or None
            Integrations per FFT window.  If None (default) the value is chosen
            automatically as the largest power-of-2 that leaves at least 4
            non-overlapping windows — balancing frequency and time resolution
            given the available number of integrations.
        """
        from scipy.signal import spectrogram as scipy_spectrogram

        called_from_report = ax is not None
        if ax is None:
            _, ax = plt.subplots()

        ints = self.integrations  # (n_int, roi_y, roi_x)
        n_int = len(ints)
        bg = np.concatenate(
            [ints[:, :, :roi_xdelta_left], ints[:, :, -roi_xdelta_right:]],
            axis=2,
        )
        pixels = bg.reshape(n_int, -1)
        pixels -= pixels.mean(axis=0, keepdims=True)

        fpi = self._frames_per_integration
        dt = (fpi * self.frame_time).to(u.second).value
        fs = 1.0 / dt

        if nperseg is None:
            # Largest power-of-2 that leaves at least 4 non-overlapping windows.
            nperseg = max(4, 2 ** int(np.log2(max(n_int // 4, 4))))

        # Compute spectrogram for each background pixel then average power
        freq, t_seg, sxx = scipy_spectrogram(
            pixels.T, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, axis=1
        )  # sxx: (n_bg_pixels, nfreqs, n_segments)
        sxx_mean = sxx.mean(axis=0)  # (nfreqs, n_segments)

        pcm = ax.pcolormesh(
            t_seg / 60,  # convert to minutes
            freq,
            np.log10(sxx_mean + 1),
            shading="nearest",
            **kwargs,
        )
        ax.set(
            yscale="log",
            xlabel="Time from Start [min]",
            ylabel="Frequency [Hz]",
        )
        plt.colorbar(pcm, ax=ax, label="log₁₀ PSD")
        if not called_from_report:
            ax.set(title=f"{self[0].header['targ_id']} {self.start_time.isot}")
        return ax

    def plot_ramp_linearity(self, ax=None, **kwargs):
        called_from_report = ax is not None
        if ax is None:
            _, ax = plt.subplots()
        t, rms = self.ramp_linearity
        t_min = (t.jd - self.start_time.jd) * 24 * 60
        ax.plot(t_min, rms, **kwargs)
        med = float(np.median(rms))
        ax.axhline(med, ls="--", color="gray", lw=1, label=f"median {med:.1f}")
        ax.legend(fontsize=8)
        ax.set(xlabel="Time from Start [min]", ylabel="Ramp Residual RMS [counts]")
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
        if np.any(dead > 0):
            ax.plot(t_min, dead, label="Dead", **kwargs)
        ax.legend()
        ax.set(
            yscale='linear',
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
            "EXPOSRES",
            "RESETS1",
            "RESETS2",
            "DROPS1",
            "DROPS2",
            "DROPS3",
            "READS",
            "GRPS",
            "INTEGRTS",
            "ROISIZEX",
            "ROISIZEY"
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
        
        # Add integration time info
        int_1_time, int_n_time = get_integration_times(hdr)
        rows.append(["INT_1_TIME", round(int_1_time, 4), "First integration time [s]"])
        rows.append(["INT_N_TIME", round(int_n_time, 4), "Subsequent integration time [s]"])

        # Add hot/dead pixel info
        _, hot = self.hot_pixels
        _, dead = self.dead_pixels
        rows.append(["HOT_MED", int(np.median(hot)), "Median hot pixels per integration"])
        rows.append(["DEAD_MED", int(np.median(dead)), "Median dead pixels per integration"])

        # Background RMS
        _, rms = self.background_rms()
        rows.append(["BG_RMS", round(float(np.median(rms)), 1), "Median background RMS [counts]"])

        # Fit PSD ~ f^(-alpha) in log-log space over the lowest third of frequencies,
        # where 1/f noise dominates.  The negative slope gives the power-law index.
        freq, psd = self.noise_psd()
        low = freq < np.percentile(freq, 33)
        if low.sum() >= 2:
            slope, _ = np.polyfit(np.log10(freq[low]), np.log10(psd[low]), 1)
            alpha = round(float(-slope), 2)
        else:
            alpha = "N/A"
        rows.append(["PSD_ALPHA", alpha, "1/f noise power-law index (PSD ~ f^-alpha)"])

        # NIRDA ramp linearity
        _, ramp_rms = self.ramp_linearity
        rows.append(["RAMP_MED", round(float(np.median(ramp_rms)), 1), "Median ramp residual RMS [counts]"])

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
            "background_rms": lambda ax=None: self.plot_background_rms(ax=ax),
            "ramp_linearity": lambda ax=None: self.plot_ramp_linearity(ax=ax),
        }

    def plot_data(self, ax=None, **kwargs):
        called_from_report = ax is not None
        if ax is None:
            _, ax = plt.subplots()
        
        use_log = kwargs.pop("log", True)
        if use_log:
            d = np.median(np.log(self.integrations), axis=0)
            max_add = 0.3
        else:
            d = np.median(self.integrations, axis=0)
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
