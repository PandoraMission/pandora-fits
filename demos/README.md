# Demos

Example notebooks that use `pandorafits` on real Pandora Level 2 files. The data are not included in this repository; set the file or folder path near the top of each notebook (the saved outputs come from the files named in each notebook).

| Notebook | What it shows | Input |
|---|---|---|
| `01_visda_reduction.ipynb` | VISDA photometry of one file: aperture flux, frame flags (pointing settle, off-target co-adds), the target's centroid against the position predicted from the pointing, decorrelation against the centroid, and the measured precision against the expected photon, background and read noise for single co-adds and binned in time. Saved outputs: HD 189733 b, 2026-09-22 01:10 UTC. | One VISDA Level 2 file (`*_VisSci_*_l2.fits`) |
| `02_nirda_reduction.ipynb` | NIRDA spectra of one file: bad pixels masked and filled from the trace profile instead of dropping whole rows, temporal outlier replacement, contaminated rows marked, median and flux-calibrated spectra, white light curve, and the measured precision per wavelength against the expected noise. Saved outputs: the same HD 189733 b observation. | One NIRDA Level 2 file (`*_InfImg_*_l2.fits`) |
| `03_visit_analysis.ipynb` | Time series of every VISDA and NIRDA file in a folder: per-file reductions as in 01 and 02, times in BJD_TDB, files grouped into visits, frame flags, detrending over each visit (keeping transits), detrended light curves per visit, background and pointing diagnostics, folded transits in both detectors and in NIRDA wavelength channels, and box depths by wavelength. Saved outputs: WASP-52 b, 4 visits. | A folder of Level 2 files and the planet's ephemeris |

The per-file methods were chosen by testing on the 2026 data: for NIRDA, masking and filling bad pixels from neighboring rows of the same frame recovers wavelengths without adding noise; for VISDA, masking pixels inside the moving image of the star increases the scatter, so flagged pixels are kept and the jitter is removed by decorrelation instead.
