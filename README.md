# `pandora-fits`

Tools to work with fits files from Pandora.

`pandora-fits` wraps `astropy.io.fits.HDUList` classes to ensure that files conform to Pandora FITS standards.

The standards are defined using excel files in the `src/pandorasat/fileformats/` folder. Changing these files will change the standards that this tool checks against.

## Pandora Detectors

Pandora has two detectors, VISDA and NIRDA. You can read more about each of these in [pandora-sat](https://github.com/PandoraMission/pandora-sat/tree/main).

## Pandora File Levels

Pandora will have the following levels of files for each detector

| Level | Description                                                  |
|-------|--------------------------------------------------------------|
| 0     | Raw data from spacecraft                                     |
| 1     | Reorganized raw data, with potential for additional keywords |
| 2     | Calibrated image data products                               |
| 3     | Spectral time-series data, ready for science.                |

## Exceptions

`pandora-fits` will throw exceptions if files are not in the correct format. This includes

- Files do not have the right number of extensions
- Extensions are not the correct type
- Header keywords have the wrong values when compared with the template

## Warnings

`pandora-fits` will log warnings if files are missing keyword headers, but those headers aren't valued in the excel spreadsheet.
