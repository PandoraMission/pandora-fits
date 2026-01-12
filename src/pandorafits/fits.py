"""Class to handle Pandora fits files"""

from datetime import timedelta

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

# import pandas as pd
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from . import logger
from .processing import ProcessingMixins
from .utils import (
    BITPIX_DICT,
    generate_random_bintable_values,
    generate_random_table_values,
    get_excel_sheet,
)

__all__ = ["FITSTemplateException", "FITSValueException", "PandoraHDUList"]

SKIPKWS = [
    "COMMENT",
    "CHECKSUM",
    "DATASUM",
]


class FITSTemplateException(Exception):
    """Custom exception for fits files not having the right shape."""

    def __init__(self, message):
        super().__init__(message)


class FITSValueException(Exception):
    """Custom exception for fits files not having the right values."""

    def __init__(self, message):
        super().__init__(message)


def _clean_header_cards(hdr: fits.Header):
    """Cleans the list of cards to ensure they have reasonable values."""
    for key in hdr:
        if key in SKIPKWS:
            continue
        if hdr[key] in ["TRUE", "True", "T"]:
            hdr[key] = True
        if hdr[key] in ["FALSE", "False", "F"]:
            hdr[key] = False

        def to_number(x):
            if isinstance(x, str):
                try:
                    return int(x.strip())
                except (ValueError, TypeError):
                    try:
                        return float(x.strip())
                    except (ValueError, TypeError):
                        return x.strip()
            return x

        hdr[key] = to_number(hdr[key])
    return hdr


class PandoraHDUList(fits.HDUList, ProcessingMixins):
    """Base class, not designed to be used. Adds mixins to the fits.HDUList object"""

    def __repr__(self):
        return f"Pandora {self.__class__.__name__}:\n\t" + "\n\t".join(
            super().__repr__()[1:-1].replace(", ", ",").split(",")
        )

    def _get_default_cards(self, extname):
        if isinstance(extname, int):
            extname = self.extension_names[extname.lower()]
        return [
            fits.Card(d.iloc[0], d.iloc[1] if d.iloc[1] != "" else d.iloc[2], d.iloc[3])
            for _, d in self.extension_headers[extname.lower()].fillna("").iterrows()
        ]

    def _get_mandetory_cards(self, extname):
        if isinstance(extname, int):
            extname = self.extension_names[extname.lower()]
        return [
            fits.Card(d.iloc[0], d.iloc[1], d.iloc[3])
            for _, d in self.extension_headers[extname.lower()].fillna("").iterrows()
        ]

    def _validate_ext_types(self):
        """Validate that the extensions have the correct types, e.g. ImageHDU, TableHDU, etc"""
        for hdu, expected_type in zip(self, self.extension_types):
            if not isinstance(hdu, getattr(fits, expected_type)):
                raise FITSTemplateException(
                    f"[EXT {hdu.header['EXTNAME']}] Data doesn't match format for {self.__class__.__name__}. "
                    + f"Expected extension type {expected_type}, got {hdu}."
                )

    def _validate_n_ext(self):
        """Validate that all the necessary extensions are present."""
        k = np.in1d(
            self.extension_names, [hdu.header["EXTNAME"].lower() for hdu in self]
        )
        for name in self.extension_names[~k]:
            if not self.structure[
                self.structure.Extension.str.lower() == name
            ].Optional.values:
                raise FITSTemplateException(
                    f"Data doesn't match format for {self.__class__.__name__}. Expected extension {name}, but none found."
                )
        self.extension_headers = {
            n: self.extension_headers[n] for n in self.extension_names[k]
        }
        self.extension_types = self.extension_types[k]
        self.extension_names = self.extension_names[k]

    def _validate_data(self):
        """Check the data in the fits file is all the right dtype, given expected `bitpix`"""
        for extname, hdu in zip(self.extension_names, self):
            if hdu.header["EXTNAME"] == "PRIMARY":
                continue
            if isinstance(hdu, fits.ImageHDU):
                expected_header = fits.Header(self._get_mandetory_cards(extname))
                expected_type, expected_type_str = BITPIX_DICT[
                    int(expected_header["bitpix"])
                ]
                if hdu.data.dtype == expected_type:
                    if int(expected_header["bitpix"]) == 32:
                        if not (int(hdu.header["BSCALE"]) == 1) & (
                            int(hdu.header["BZERO"]) == 2**31
                        ):
                            raise FITSTemplateException(
                                f"[EXT {hdu.header['EXTNAME']}] Data doesn't match format for {self.__class__.__name__}."
                                f" Expected data type of np.uint32, got {hdu.data.dtype}"
                            )
                    if int(expected_header["bitpix"]) == -64:
                        continue
                    else:
                        raise FITSTemplateException(
                            f"[EXT {hdu.header['EXTNAME']}] Data doesn't match format for {self.__class__.__name__}. "
                            f"Expected data of type {expected_type} ({expected_type_str}), got {hdu.data.dtype}"
                        )

    def _validate_mandetory_headers(self, warn=False):
        """Validate the data contains mandetory header cards."""
        for extname, hdu in zip(self.extension_names, self):
            hdr = _clean_header_cards(hdu.header)
            expected_header = _clean_header_cards(
                fits.Header(self._get_mandetory_cards(extname))
            )
            for key in expected_header:
                if key in SKIPKWS:
                    continue
                # fill missing cards
                if key not in hdr:
                    if warn:
                        hdr[key] = expected_header[key]
                        logger.warning(
                            f"[EXT {hdr['EXTNAME']}] Key {key} expected in extension `{extname}` but not found. Added this key."
                        )
                    else:
                        raise FITSValueException(
                            f"[EXT {hdr['EXTNAME']}] {key} header keyword expected for {self.__class__.__name__} in extension `{extname}`,"
                            " but not found in data provided."
                        )
                # check mandetory cards have the correct values
                if expected_header[key] not in ["", None, np.nan]:
                    if hdr[key] != expected_header[key]:
                        if isinstance(expected_header[key], bool):
                            continue
                        raise FITSValueException(
                            f"[EXT {hdr['EXTNAME']}] {key} expected to have value of {expected_header[key]}, but has value {hdr[key]}."
                        )

    def _validate_no_extra_keywords(self, warn=False):
        """Validate there are no additional keywords in the file"""
        for extname, hdu in zip(self.extension_names, self):
            hdr = _clean_header_cards(hdu.header)
            expected_header = _clean_header_cards(
                fits.Header(self._get_mandetory_cards(extname))
            )
            for key in hdr:
                if key in SKIPKWS:
                    continue
                if key not in expected_header:
                    if warn:
                        hdr.pop(key)
                        logger.warning(
                            f"[EXT {hdr['EXTNAME']}] {key} found in extension `{extname}` header but not expected. Removed this key."
                        )
                    else:
                        raise FITSTemplateException(
                            f"[EXT {hdr['EXTNAME']}] {key} header keyword is not expected for {self.__class__.__name__}"
                            + f" in extension `{extname}`."
                        )

    def _get_dummy_hdus(self):
        hdulist = []
        for extname, exttype in zip(self.extension_names, self.extension_types):
            cards = self._get_default_cards(extname)
            hdr = _clean_header_cards(fits.Header(cards))
            data = None
            if exttype == "PrimaryHDU":
                hdu = fits.PrimaryHDU(header=hdr)
            elif exttype == "CompImageHDU":
                shape = tuple(
                    [
                        int(hdr[f"NAXIS{naxis}"])
                        for naxis in np.arange(1, hdr["NAXIS"] + 1)[::-1]
                    ]
                )
                if "BSCALE" in hdr:
                    if (int(hdr["BSCALE"]) == 1) and (int(hdr["BZERO"]) == 2**15):
                        data = np.ones(shape, dtype=np.uint16)
                    else:
                        raise FITSValueException(
                            f"[EXT {hdr['EXTNAME']}] Can not parse data type"
                        )
                else:
                    data = np.ones(shape, dtype=BITPIX_DICT[hdr["BITPIX"]][0])
                hdu = fits.CompImageHDU(header=hdr, data=data)
            elif exttype == "ImageHDU":
                shape = tuple(
                    [
                        int(hdr[f"NAXIS{naxis}"])
                        for naxis in np.arange(1, hdr["NAXIS"] + 1)[::-1]
                    ]
                )
                if "BSCALE" in hdr:
                    if (int(hdr["BSCALE"]) == 1) and (int(hdr["BZERO"]) == 2**31):
                        data = np.ones(shape, dtype=np.uint32)
                    else:
                        raise FITSValueException(
                            f"[EXT {hdr['EXTNAME']}] Can not parse data type"
                        )
                else:
                    data = np.ones(shape, dtype=BITPIX_DICT[hdr["BITPIX"]][0])
                hdu = fits.ImageHDU(header=hdr, data=data)
            elif exttype == "TableHDU":
                ncolumns = len(
                    [c.keyword for c in cards if c.keyword.startswith("TTYPE")]
                )
                columns = [
                    fits.Column(
                        name=hdr[f"TTYPE{idx}"],
                        format=hdr[f"TFORM{idx}"],
                        unit=hdr[f"TUNIT{idx}"] if f"TUNIT{idx}" in hdr else "",
                        array=(
                            generate_random_table_values(
                                hdr[f"TFORM{idx}"], hdr["NAXIS2"]
                            )
                            if hdr["NAXIS2"] != ""
                            else None
                        ),
                    )
                    for idx in np.arange(1, ncolumns + 1)
                ]

                hdu = fits.TableHDU.from_columns(columns, header=hdr)
            elif exttype == "BinTableHDU":
                ncolumns = len(
                    [c.keyword for c in cards if c.keyword.startswith("TTYPE")]
                )
                columns = [
                    fits.Column(
                        name=hdr[f"TTYPE{idx}"],
                        format=hdr[f"TFORM{idx}"],
                        unit=hdr[f"TUNIT{idx}"] if f"TUNIT{idx}" in hdr else "",
                        array=(
                            generate_random_bintable_values(
                                hdr[f"TFORM{idx}"], hdr["NAXIS2"]
                            )
                            if hdr["NAXIS2"] != ""
                            else None
                        ),
                    )
                    for idx in np.arange(1, ncolumns + 1)
                ]

                hdu = fits.BinTableHDU.from_columns(columns, header=hdr)
            else:
                raise FITSValueException(
                    f"[EXT {hdr['EXTNAME']}] No extension type {exttype}."
                )
            cards = self._get_mandetory_cards(extname)
            _ = [hdu.header.append(card) for card in cards if card[0] not in hdu.header]
            hdulist.append(hdu)
        return fits.HDUList(hdulist)

    def __init__(self, file=None, validate=True, warn=True):
        self.warn = warn
        self.structure = get_excel_sheet(self.filename, 0)
        # load in header formats
        self.extension_headers = [
            get_excel_sheet(self.filename, idx + 1)
            for idx in range(len(self.structure))
        ]
        # convert to dictionary
        self.extension_headers = {
            h["Fixed Value"][h.Name.isin(["EXTNAME"])].values[0].lower(): h
            for h in self.extension_headers
        }
        # remove optional extensions
        self.structure = self.structure[
            self.structure["Extension"].str.lower().isin(self.extension_headers.keys())
        ]
        self.extension_names = np.asarray(self.structure.Extension.str.lower().values)
        self.extension_types = np.asarray(self.structure.Type.values)

        if file is None:
            logger.warning("Creating a dummy file.")
            hdulist = self._get_dummy_hdus()
            super().__init__(hdulist)
        elif isinstance(file, str):
            super().__init__(fits.open(file))
        elif isinstance(file, fits.HDUList):
            super().__init__(file)
        if validate:
            self._validate_n_ext()
            self._validate_ext_types()
            self._validate_data()
            self._validate_mandetory_headers(warn=self.warn)
            self._validate_no_extra_keywords(warn=self.warn)

    def writeto(
        self,
        fileobj,
        output_verify="exception",
        overwrite=False,
        checksum=False,
    ):
        fits.HDUList(self).writeto(
            fileobj=fileobj,
            output_verify=output_verify,
            overwrite=overwrite,
            checksum=checksum,
        )

    def copy(self):
        return self.__class__(super().copy())

    @property
    def targ(self):
        if "TARG_RA" in self[0].header:
            return SkyCoord(
                self[0].header["TARG_RA"], self[0].header["TARG_DEC"], unit="deg"
            )
        else:
            return None

    @property
    def wcs(self):
        if np.any(["WCSAXES" in self[idx].header for idx in range(len(self))]):
            return [
                WCS(self[idx].header[3:])
                for idx in range(len(self))
                if "WCSAXES" in self[idx].header
            ][0]
        else:
            return None

    @property
    def start_time(self):
        """Given Pandora HDUList obtains the detector time in TAI."""
        time = (
            Time("2000-01-01 12:00:00", scale="tai")
            + timedelta(
                seconds=self[0].header["CORSTIME"],
                milliseconds=self[0].header["FINETIME"] / 1e6,
            )
        ).utc
        return time

    @property
    def sequence_start_time(self):
        """Given Pandora HDUList obtains the detector time in TAI."""
        if "SEQSTART" in self[0].header:
            return Time(self[0].header["SEQSTART"], format="jd")
        else:
            return self.start_time

    @property
    def pixel_coordinates(self):
        hdr = self[0].header
        R, C = np.mgrid[
            hdr["ROISTRTY"] : hdr["ROISTRTY"] + hdr["ROISIZEY"],
            hdr["ROISTRTX"] : hdr["ROISTRTX"] + hdr["ROISIZEX"],
        ]
        return R, C

    @property
    def row(self):
        hdr = self[0].header
        return np.arange(hdr["ROISTRTY"], hdr["ROISTRTY"] + hdr["ROISIZEY"])

    @property
    def column(self):
        hdr = self[0].header
        return np.arange(hdr["ROISTRTX"], hdr["ROISTRTX"] + hdr["ROISIZEX"])

    @property
    def frame_time(self):
        if "FRMTIME" in self[0].header:
            if self[0].header["GRPSAVGD"] == 0:
                return (u.millisecond * self[0].header["FRMTIME"]).to(u.second)
            else:
                return (
                    u.millisecond * self[0].header["FRMTIME"] * self[0].header["READS"]
                ).to(u.second)
        elif "EXPTIMEU" in self[0].header:
            return (
                u.microsecond
                * self[0].header["EXPTIMEU"]
                * self[0].header["FRMSCLCT"]
                / self[1].header["NAXIS3"]
            ).to(u.second)
        elif "EXPTIME" in self[0].header:
            return (
                u.microsecond
                * self[0].header["EXPTIME"]
                * self[0].header["FRMSCLCT"]
                / self[1].header["NAXIS3"]
            ).to(u.second)

    @property
    def nframes(self):
        return self[1].header[f"NAXIS{self[1].header['NAXIS']}"]

    @property
    def ncoadds(self):
        if "FRMPCOAD" in self[0].header:
            return self[0].header["FRMPCOAD"]
        elif self[0].header["INSTRMNT"] == "VISDA":
            return 1
        if self[0].header["GRPSAVGD"] == 0:
            return 1
        else:
            return self[0].header["GRPS"]

    @property
    def end_time(self):
        return self.start_time + self.nframes * self.frame_time

    def time(self):
        dt = timedelta(seconds=self.frame_time.to(u.second).value)
        return (self.start_time + (np.arange(self.nframes) * dt)).jd
