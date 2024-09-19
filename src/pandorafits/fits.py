"""Class to handle Pandora fits files"""

from . import logger
from . import BITPIX_DICT, FORMATSDIR
import numpy as np
from astropy.io import fits
import pandas as pd


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

    def _get_expected_cards(self, index):
        return [
            fits.Card(*d.fillna("").values)
            for _, d in self.header_formats[index].iterrows()
        ]

    def _get_dummy_hdus(self):
        hdulist = []
        for idx, d in self.extension_types.iterrows():
            cards = self._get_expected_cards(idx)
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
                data = np.ones(shape, dtype=BITPIX_DICT[hdr["BITPIX"]])
                hdu = fits.ImageHDU(header=hdr, data=data)
            elif d.Type == "TableHDU":
                ncolumns = len(
                    [c.keyword for c in cards if c.keyword.startswith("TTYPE")]
                )
                coldefs = fits.ColDefs(
                    [
                        fits.Column(
                            name=hdr[f"TTYPE{idx}"],
                            format=hdr[f"TFORM{idx}"],
                            unit=f"TUNIT{idx}",
                        )
                        for idx in np.arange(1, ncolumns + 1)
                    ]
                )
                hdu = fits.TableHDU.from_columns(coldefs, header=hdr)
            hdulist.append(hdu)
        return hdulist

    def _validate_ext_types(self):
        """Validate that the extensions have the correct types, e.g. ImageHDU, TableHDU, etc"""
        if not len(self) == self.nextension:
            raise FITSTemplateException(
                f"Expected {self.nextension} extensions, got {len(self)}."
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
            expected_header = fits.Header(self._get_expected_cards(idx))
            for key in expected_header:
                if not (key in hdr):
                    logger.warning(f"Key {key} expected, but not found.")
                    continue
                if expected_header[key] in ["", None, np.nan]:
                    logger.warning(f"{key} header key missing from ext {idx}.")
                else:
                    if hdr[key] != expected_header[key]:
                        if key not in [
                            "NAXIS1",
                            "NAXIS2",
                            "NAXIS3",
                            "NAXIS4",
                            "NAXIS5",
                        ]:
                            FITSValueException(
                                f"{key} expected to have value of {expected_header[key]}, but has value {hdr[key]}."
                            )

    def _validate_data(self):
        """Check the data in the fits file is all the right dtype, given expected `bitpix`"""
        for idx, hdu in enumerate(self):
            if hdu.header["EXTNAME"] == "PRIMARY":
                continue
            if isinstance(hdu, fits.ImageHDU):
                expected_header = fits.Header(self._get_expected_cards(idx))
                expected_type = BITPIX_DICT[expected_header["bitpix"]]
                if not hdu.data.dtype == expected_type:
                    raise FITSTemplateException(
                        f"Expected data of type {expected_type}, got {hdu.data.dtype}"
                    )

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
        self.nextension = len(self.header_formats)
        self.validate()

    def writeto(self, *args, **kwargs):
        """Write to file

        Here we will add in some functionality to add in keywords on write that express the history somehow, and check the file names?
        """
        fits.HDUList(self).writeto(*args, **kwargs)


class NIRDA0HDUList(PandoraHDUList):
    """NIRDA Level 0 File Type"""

    def __init__(self, file=None):
        """
        This class will read a file passed to it using astropy fits.
        After loading it will validate that the file is compliant with
        the NIRDA Level 0 file standards specified in the `fileformats`
        folder. This object subclasses the fits.HDUList object, and
        maintains all its class methods.

        Parameters:
        -----------
        file: None, str, fits.HDUList
            The file to load
        """
        self.header_formats = [
            pd.read_excel(FORMATSDIR + "nirda-headers-level0.xlsx", idx)
            for idx in range(2)
        ]
        self.extension_types = pd.read_excel(
            FORMATSDIR + "nirda-extension-types-level0.xlsx"
        )
        super().__init__(file=file)


class NIRDA2HDUList(PandoraHDUList):
    def __init__(self, file=None):
        self.header_formats = [
            pd.read_excel(FORMATSDIR + "nirda-headers-level2.xlsx", idx)
            for idx in range(6)
        ]
        self.extension_types = pd.read_excel(
            FORMATSDIR + "nirda-extension-types-level2.xlsx"
        )
        super().__init__(file=file)


class VISDA0HDUList(PandoraHDUList):
    def __init__(self, file=None):
        self.header_formats = [
            pd.read_excel(FORMATSDIR + "visda-headers-level0.xlsx", idx)
            for idx in range(3)
        ]
        self.extension_types = pd.read_excel(
            FORMATSDIR + "visda-extension-types-level0.xlsx"
        )
        super().__init__(file=file)


class VISDA2HDUList(PandoraHDUList):
    def __init__(self, file=None, nROIs: int = 9):
        self.header_formats = [
            pd.read_excel(FORMATSDIR + "visda-headers-level2.xlsx", 0),
            *[
                pd.read_excel(FORMATSDIR + "visda-headers-level2.xlsx", 1)
                for _ in range(1, nROIs + 1)
            ],
            *[
                pd.read_excel(FORMATSDIR + "visda-headers-level2.xlsx", idx)
                for idx in range(2, 5)
            ],
        ]
        self.extension_types = pd.read_excel(
            FORMATSDIR + "visda-extension-types-level2.xlsx"
        )
        self.header_formats[1].loc[
            self.header_formats[1].Name == "EXTNAME", "Value"
        ] = "TARGET"
        for idx in np.arange(2, nROIs + 2):
            self.header_formats[idx].loc[
                self.header_formats[1].Name == "EXTNAME", "Value"
            ] = f"STAR{idx - 1:03}"
        super().__init__(file=file)
