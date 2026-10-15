# Demos

Example notebooks that use `pandorafits` on real Pandora files. The data are not included in this repository; set the file path at the top of each notebook to a file on your machine.

| Notebook | What it shows | Input |
|---|---|---|
| `nirda_spectrum_extraction.ipynb` | Extracting a time series of NIRDA spectra from a Level 2 file: bad pixels masked and filled from the trace profile instead of dropping whole rows, temporal outlier replacement, contaminated rows marked, median and flux-calibrated spectra, white light curve, and the measured precision per wavelength compared with the expected photon, background and read noise. | One NIRDA Level 2 file (`*_InfImg_*_l2.fits`) |
