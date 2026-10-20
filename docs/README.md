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

## Timestamp conventions

L1 and L2 `TIME` extensions contain Julian Dates (`FRAME = 'JD'`) in UTC
(`TIMESYS = 'UTC'`). Spacecraft counters are measured from
2000-01-01 00:00:00 TAI and converted to UTC. `EXPTIME` extensions use seconds.

- **VISDA science and FFI:** `TIME` is the frame start. Science frames contain
  `FRMPCOAD` coadded reads; FFI frames contain one read. `EXPTIME` is the frame
  duration, so the end is `TIME + EXPTIME / 86400` when using the stored arrays.
  The VISDA primary-header `EXPTIME` instead gives the read duration in
  microseconds.
- **NIRDA L1:** `TIME` follows the detector-read sequence, including reset and
  dropped-frame intervals. `EXPTIME` is cumulative ramp exposure through each
  read, not a duration starting at its `TIME`. For onboard-averaged groups,
  `TIME` uses the first read's timestamp and `EXPTIME` is averaged over the reads.
- **NIRDA L2:** Adjacent reads (or groups) are differenced. Each product retains
  the later L1 read/group's `TIME`; its `EXPTIME` is the difference between the
  corresponding cumulative L1 exposures. This mapping does not establish that
  L2 `TIME` is the physical exposure start or that `TIME + EXPTIME / 86400`
  gives its end.

Older files may lack `TIMESYS`. Constructor validation with the default
`warn=True` adds the required UTC card in memory; strict validation
(`warn=False`) rejects a missing card. Older package versions may reject the new
card under strict validation.

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
from pandorafits.nirda import NIRDALevel0HDUList
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

If you need to run the pipeline and use the databasing features you will need to add the optional extras. 

```
pip install "pandorafits[database]" --upgrade
```

Note if you're working on a windows machine this install might be harder and you should open an issue if you encounter any problems.
