[![pypi-badge](https://img.shields.io/pypi/v/pandorafits.svg?color=blue)](https://pypi.python.org/pypi/pandorafits)
<a href="https://github.com/pandoramission/pandora-fits/actions/workflows/tests.yml"><img src="https://github.com/pandoramission/pandora-fits/workflows/pytest/badge.svg" alt="Test status"/></a>

# `pandora-fits`

Tools to work with fits files from Pandora.

`pandora-fits` wraps `astropy.io.fits.HDUList` classes to ensure that files conform to Pandora FITS standards.

The standards are defined using excel files in the `src/pandorasat/formats/` folder. Changing these files will change the standards that this tool checks against.

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

## Usage

You should treat the `pandora-fits` objects as though they were `astropy.io.fits.HDUList` objects. There is one per detector, per file level.

First you can import the correct `HDUList` object. Note that I am using the logger and setting the logger to the level `"ERROR"`.

```python
from pandorafits.fits import NIRDALevel0HDUList
from pandorafits import logger
logger.setLevel("ERROR")
```

We can create a dummy file by passing nothing to the object

```python
hdulist = NIRDALevel0HDUList()
hdulist.info()
```

    Filename: (No file associated with this HDUList)
    No.    Name      Ver    Type      Cards   Dimensions   Format
      0  PRIMARY       1 PrimaryHDU      35   ()      
      1  SCIENCE       1 ImageHDU        12   (80, 400, 10)   int16   

This initializes an "empty" file that is compliant with the Pandora scheme. There are several logger "warning" messages that will state that many of the header keywords are not set.

We can write to a file

```python
hdulist.writeto("test.fits", overwrite=True)
```

We can also read in a file

```python
hdulist = NIRDALevel0HDUList("test.fits")
```

Finally we can read in an existing HDUList, e.g.

```python
hdulist = fits.HDUList("test.fits")
NIRDALevel0HDUList(hdulist)
```

## Installation

You can install with a git clone, or via PyPI using the command below. Make sure to update to the most recent version.

```
pip install pandorafits --upgrade
```
