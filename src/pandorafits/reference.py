"""module to deal with reference product loading and dummy files."""

import numpy as np
from astropy.io import fits
from astropy.time import Time
from . import __version__


__all__ = [
    "create_visda_dummy_bad_pixel_map",
    "create_nirda_dummy_bad_pixel_map",
    "create_visda_dummy_flat",
    "create_nirda_dummy_flat",
]


def create_visda_dummy_bad_pixel_map():
    """Creates a dummy file that is a placeholder for a bad pixel map on the VISDA."""
    quality = np.zeros((2048, 2048), dtype=np.uint16)
    for idx in range(16):
        quality[idx * 10 : (idx + 1) * 10, idx * 10 : (idx + 1) * 10] = 2**idx

    hdr0 = fits.Header(
        [
            ("SIMDATA", True, "simulated data"),
            ("SCIDATA", False, "science data"),
            ("TELESCOP", "NASA Pandora", "telescope"),
            ("CAMERAID", "PcoCam", "ID of camera used in acquisition"),
            ("INSTRMNT", "VISDA", "instrument"),
            ("CREATOR", "Pandora DPC", "creator of this product"),
            ("CRSOFTV", __version__, "creator software version"),
            ("FILEV", "0.1.0", "file version"),
            ("DATE", Time.now().isot, "creation date"),
            ("COMMENT", "This file has been created as a place holder for a RDP"),
        ]
    )

    hdr1 = fits.Header(
        [
            ("EXTNAME", "BAD_PIX", "name of extension"),
            ("0", "NO ISSUE", "bit definition"),
            ("1", "Placeholder", "bit definition"),
            ("2", "Placeholder", "bit definition"),
            ("3", "Placeholder", "bit definition"),
            ("4", "Placeholder", "bit definition"),
            ("5", "Placeholder", "bit definition"),
            ("6", "Placeholder", "bit definition"),
            ("7", "Placeholder", "bit definition"),
            ("8", "Placeholder", "bit definition"),
            ("9", "Placeholder", "bit definition"),
            ("10", "Placeholder", "bit definition"),
            ("11", "Placeholder", "bit definition"),
            ("12", "Placeholder", "bit definition"),
            ("13", "Placeholder", "bit definition"),
            ("14", "Placeholder", "bit definition"),
            ("15", "Placeholder", "bit definition"),
        ]
    )
    hdulist = fits.HDUList(
        [fits.PrimaryHDU(header=hdr0), fits.CompImageHDU(quality, header=hdr1)]
    )
    return hdulist


def create_nirda_dummy_bad_pixel_map():
    """Creates a dummy file that is a placeholder for a bad pixel map on the NIRDA."""
    quality = np.zeros((2048, 2048), dtype=np.uint16)
    for idx in range(16):
        quality[idx * 10 : (idx + 1) * 10, idx * 10 : (idx + 1) * 10] = 2**idx

    hdr0 = fits.Header(
        [
            ("SIMDATA", True, "simulated data"),
            ("SCIDATA", False, "science data"),
            ("TELESCOP", "NASA Pandora", "telescope"),
            ("CAMERAID", "H2rgCam", "ID of camera used in acquisition"),
            ("INSTRMNT", "NIRDA", "instrument"),
            ("CREATOR", "Pandora DPC", "creator of this product"),
            ("CRSOFTV", __version__, "creator software version"),
            ("FILEV", "0.1.0", "file version"),
            ("DATE", Time.now().isot, "creation date"),
            ("COMMENT", "This file has been created as a place holder for a RDP"),
        ]
    )

    hdr1 = fits.Header(
        [
            ("EXTNAME", "BAD_PIX", "name of extension"),
            ("0", "NO ISSUE", "bit definition"),
            ("1", "Placeholder", "bit definition"),
            ("2", "Placeholder", "bit definition"),
            ("3", "Placeholder", "bit definition"),
            ("4", "Placeholder", "bit definition"),
            ("5", "Placeholder", "bit definition"),
            ("6", "Placeholder", "bit definition"),
            ("7", "Placeholder", "bit definition"),
            ("8", "Placeholder", "bit definition"),
            ("9", "Placeholder", "bit definition"),
            ("10", "Placeholder", "bit definition"),
            ("11", "Placeholder", "bit definition"),
            ("12", "Placeholder", "bit definition"),
            ("13", "Placeholder", "bit definition"),
            ("14", "Placeholder", "bit definition"),
            ("15", "Placeholder", "bit definition"),
        ]
    )
    hdulist = fits.HDUList(
        [fits.PrimaryHDU(header=hdr0), fits.CompImageHDU(quality, header=hdr1)]
    )
    return hdulist


def create_visda_dummy_flat():
    """Creates a dummy file that is a placeholder for a flat on the VISDA."""
    flat = np.ones((2048, 2048), dtype=np.float32)

    hdr0 = fits.Header(
        [
            ("SIMDATA", True, "simulated data"),
            ("SCIDATA", False, "science data"),
            ("TELESCOP", "NASA Pandora", "telescope"),
            ("CAMERAID", "PcoCam", "ID of camera used in acquisition"),
            ("INSTRMNT", "VISDA", "instrument"),
            ("CREATOR", "Pandora DPC", "creator of this product"),
            ("CRSOFTV", __version__, "creator software version"),
            ("FILEV", "0.1.0", "file version"),
            ("DATE", Time.now().isot, "creation date"),
            ("COMMENT", "This file has been created as a place holder for a RDP"),
        ]
    )

    hdr1 = fits.Header(
        [
            ("EXTNAME", "FLAT", "name of extension"),
        ]
    )
    hdulist = fits.HDUList(
        [fits.PrimaryHDU(header=hdr0), fits.CompImageHDU(flat, header=hdr1)]
    )
    return hdulist


def create_nirda_dummy_flat():
    """Creates a dummy file that is a placeholder for a flat on the VISDA."""
    flat = np.ones((2048, 2048), dtype=np.float32)

    hdr0 = fits.Header(
        [
            ("SIMDATA", True, "simulated data"),
            ("SCIDATA", False, "science data"),
            ("TELESCOP", "NASA Pandora", "telescope"),
            ("CAMERAID", "H2rgCam", "ID of camera used in acquisition"),
            ("INSTRMNT", "NIRDA", "instrument"),
            ("CREATOR", "Pandora DPC", "creator of this product"),
            ("CRSOFTV", __version__, "creator software version"),
            ("FILEV", "0.1.0", "file version"),
            ("DATE", Time.now().isot, "creation date"),
            ("COMMENT", "This file has been created as a place holder for a RDP"),
        ]
    )

    hdr1 = fits.Header(
        [
            ("EXTNAME", "FLAT", "name of extension"),
        ]
    )
    hdulist = fits.HDUList(
        [fits.PrimaryHDU(header=hdr0), fits.CompImageHDU(flat, header=hdr1)]
    )
    return hdulist


def create_dummy_reference_products():
    pass
