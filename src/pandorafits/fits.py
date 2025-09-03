"""Class to handle Pandora fits files"""

import numpy as np
import pandas as pd
from astropy.io import fits

from . import FORMATSDIR, logger
from .utils import BITPIX_DICT, generate_random_table_values

__all__ = [
    "PandoraHDUList",
    "EngineeringLevel0HDUList",
    "Level3HDUList",
]


class FITSTemplateException(Exception):
    """Custom exception for fits files not having the right shape."""

    def __init__(self, message):
        super().__init__(message)


class FITSValueException(Exception):
    """Custom exception for fits files not having the right values."""

    def __init__(self, message):
        super().__init__(message)


class FITSHandlerMixins(object):
    """Mixins to verify fits objects have the expected formats"""

    def _get_dummy_cards(self, index):
        return [
            fits.Card(d.iloc[0], d.iloc[1] if d.iloc[1] != "" else d.iloc[2], d.iloc[3])
            for _, d in self.header_formats[index].fillna("").iterrows()
        ]

    def _get_mandetory_cards(self, index):
        return [
            fits.Card(d.iloc[0], d.iloc[1], d.iloc[3])
            for _, d in self.header_formats[index].fillna("").iterrows()
        ]

        return [
            fits.Card(*d.fillna("").values)
            for _, d in self.header_formats[index].iterrows()
        ]

    def _get_dummy_hdus(self, extensions=None):
        hdulist = []
        if extensions is None:
            k = np.ones(len(self.extension_types), bool)
        else:
            k = np.in1d(np.arange(len(self.extension_types)), extensions)
        for idx, d in self.extension_types[k].iterrows():
            cards = self._get_dummy_cards(idx)
            hdr = fits.Header(cards)
            data = None
            if d.Type == "PrimaryHDU":
                hdu = fits.PrimaryHDU(header=hdr)
            elif d.Type == "ImageHDU":
                shape = tuple(
                    [
                        hdr[f"NAXIS{naxis}"]
                        for naxis in np.arange(1, hdr["NAXIS"] + 1)[::-1]
                    ]
                )
                if "BSCALE" in hdr:
                    if (hdr["BSCALE"] == 1) and (hdr["BZERO"] == 2**31):
                        data = np.ones(shape, dtype=np.uint32)
                    else:
                        raise FITSValueException("Can not parse data type")
                else:
                    data = np.ones(shape, dtype=BITPIX_DICT[hdr["BITPIX"]][0])
                hdu = fits.ImageHDU(header=hdr, data=data)
            elif d.Type == "TableHDU":
                ncolumns = len(
                    [c.keyword for c in cards if c.keyword.startswith("TTYPE")]
                )
                columns = [
                    fits.Column(
                        name=hdr[f"TTYPE{idx}"],
                        format=hdr[f"TFORM{idx}"],
                        unit=hdr[f"TUNIT{idx}"] if f"TUNIT{idx}" in hdr else "",
                        array=generate_random_table_values(
                            hdr[f"TFORM{idx}"], hdr["NAXIS2"]
                        )
                        if hdr["NAXIS2"] != ""
                        else None,
                    )
                    for idx in np.arange(1, ncolumns + 1)
                ]

                hdu = fits.TableHDU.from_columns(columns, header=hdr)
            hdulist.append(hdu)
        return hdulist

    def _validate_ext_types(self):
        """Validate that the extensions have the correct types, e.g. ImageHDU, TableHDU, etc"""
        if not len(self) >= self.nmin_extension:
            raise FITSTemplateException(
                f"Expected {self.nmin_extension} extensions at minimum, got {len(self)}."
            )
        for hdu, expected_type in zip(self, self.extension_types.Type.values):
            if not isinstance(hdu, getattr(fits, expected_type)):
                raise FITSTemplateException(
                    f"Expected extension type {expected_type}, got {hdu}."
                )

    def _validate_headers(self):
        """Validate the extensions have the right header keywords"""
        for idx, hdu in enumerate(self):
            hdr = hdu.header
            expected_header = fits.Header(self._get_mandetory_cards(idx))
            for key in expected_header:
                if hdr[key] in ["TRUE", "True", "T"]:
                    hdr[key] = True
                if hdr[key] in ["FALSE", "False", "F"]:
                    hdr[key] = False
                if expected_header[key] in ["TRUE", "True", "T"]:
                    expected_header[key] = True
                if expected_header[key] in ["FALSE", "False", "F"]:
                    expected_header[key] = False
                if not (key in hdr):
                    logger.warning(f"Key {key} expected, but not found.")
                    continue
                if hdr[key] in ["", None, np.nan]:
                    logger.warning(f"{key} header key missing from ext {idx}.")
                if expected_header[key] not in ["", None, np.nan]:
                    if hdr[key] != expected_header[key]:
                        if isinstance(expected_header[key], bool):
                            continue
                        raise FITSValueException(
                            f"{key} expected to have value of {expected_header[key]}, but has value {hdr[key]}."
                        )
            for key in hdr:
                if key not in expected_header:
                    raise FITSTemplateException(
                        f"{key} is not an expected header value."
                    )

    def _validate_data(self):
        """Check the data in the fits file is all the right dtype, given expected `bitpix`"""
        for idx, hdu in enumerate(self):
            if hdu.header["EXTNAME"] == "PRIMARY":
                continue
            if isinstance(hdu, fits.ImageHDU):
                expected_header = fits.Header(self._get_mandetory_cards(idx))
                expected_type, expected_type_str = BITPIX_DICT[
                    expected_header["bitpix"]
                ]
                if not hdu.data.dtype == expected_type:
                    if expected_header["bitpix"] == 32:
                        if not (hdu.header["BSCALE"] == 1) & (
                            hdu.header["BZERO"] == 2**31
                        ):
                            raise FITSTemplateException(
                                f"Expected data type of np.uint32, got {hdu.data.dtype}"
                            )
                    else:
                        raise FITSTemplateException(
                            f"Expected data of type {expected_type} ({expected_type_str}), got {hdu.data.dtype}"
                        )

    def _validate_optional_extensions(self):
        """Some extensions appear to be optional. So that we have data uniformity, we'll create dummy versions of optional extensions."""

    def validate(self):
        """Validate all aspects of the file"""
        self._validate_ext_types()
        self._validate_headers()
        self._validate_data()


class PandoraHDUList(fits.HDUList, FITSHandlerMixins):
    """Base class, not designed to be used. Adds mixins to the fits.HDUList object"""

    def __init__(self, file=None):
        if file is None:
            super().__init__(self._get_dummy_hdus())
        elif isinstance(file, str):
            super().__init__(fits.open(file))
        elif isinstance(file, fits.HDUList):
            super().__init__(file)
        self.nmin_extension = (~self.extension_types.Optional.values).sum()
        self.nextension = len(self.header_formats)
        self.extnames = np.asarray(
            [
                h["Fixed Value"][h.Name.isin(["EXTNAME"])].values[0]
                for h in self.header_formats
            ]
        )
        self.validate()

    def writeto(self, *args, **kwargs):
        """Write to file

        Here we will add in some functionality to add in keywords on write that express the history somehow, and check the file names?
        """
        fits.HDUList(self).writeto(*args, **kwargs)


class EngineeringLevel0HDUList(PandoraHDUList):
    """Engineering Level 0 File Type"""

    def __init__(self, file=None):
        self.header_formats = [
            pd.read_excel(FORMATSDIR + "engineering/level0-headers.xlsx", idx)
            for idx in range(8)
        ]
        self.extension_types = pd.read_excel(
            FORMATSDIR + "engineering/level0-extension-types.xlsx"
        )
        super().__init__(file=file)


class Level3HDUList(PandoraHDUList):
    """Level 3 File Type"""

    def __init__(self, file=None):
        self.header_formats = [
            pd.read_excel(FORMATSDIR + "level3/level3-headers.xlsx", idx)
            for idx in range(6)
        ]
        self.extension_types = pd.read_excel(
            FORMATSDIR + "level3/level3-extension-types.xlsx"
        )
        super().__init__(file=file)
