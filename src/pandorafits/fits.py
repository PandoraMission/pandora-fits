"""Class to handle Pandora fits files"""

# Standard library
import warnings
from functools import cached_property

# Third-party
import astropy.units as u
import gaiaoffline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandoraspacecraft as psc
from astropy.coordinates import Distance, SkyCoord

# import pandas as pd
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from . import logger, ps
from .processing import ProcessingMixins
from .utils import (
    BITPIX_DICT,
    convert_time,
    generate_random_bintable_values,
    generate_random_table_values,
    get_excel_sheet,
    get_exposure_time,
    get_read_time,
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
        if key == "":
            continue
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
            fits.Card(
                d.iloc[0],
                d.iloc[1] if d.iloc[1] != "" else d.iloc[2],
                d.iloc[3],
            )
            for _, d in self.extension_headers[extname.lower()]
            .fillna("")
            .iterrows()
        ]

    def _get_mandetory_cards(self, extname):
        if isinstance(extname, int):
            extname = self.extension_names[extname.lower()]
        return [
            fits.Card(d.iloc[0], d.iloc[1], d.iloc[3])
            for _, d in self.extension_headers[extname.lower()]
            .fillna("")
            .iterrows()
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
        try:
            k = np.isin(
                self.extension_names,
                [hdu.header["EXTNAME"].lower() for hdu in self],
            )
        except AttributeError:
            # Newer versions of numpy removed in1d. Keeping for backward compat.
            k = np.isin(
                self.extension_names,
                [hdu.header["EXTNAME"].lower() for hdu in self],
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
                expected_header = fits.Header(
                    self._get_mandetory_cards(extname)
                )
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
                    if (int(expected_header["bitpix"]) == -64) | (
                        int(expected_header["bitpix"]) == 64
                    ):
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
        for extname, exttype in zip(
            self.extension_names, self.extension_types
        ):
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
                    if (int(hdr["BSCALE"]) == 1) and (
                        int(hdr["BZERO"]) == 2**15
                    ):
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
                    if (int(hdr["BSCALE"]) == 1) and (
                        int(hdr["BZERO"]) == 2**31
                    ):
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
                        unit=(
                            hdr[f"TUNIT{idx}"] if f"TUNIT{idx}" in hdr else ""
                        ),
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
                        unit=(
                            hdr[f"TUNIT{idx}"] if f"TUNIT{idx}" in hdr else ""
                        ),
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
            _ = [
                hdu.header.append(card)
                for card in cards
                if card[0] not in hdu.header
            ]
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
            self.structure["Extension"]
            .str.lower()
            .isin(self.extension_headers.keys())
        ]
        self.extension_names = np.asarray(
            self.structure.Extension.str.lower().values
        )
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
    def targ_id(self):
        return self[0].header["TARG_ID"]

    @property
    def targ_ra(self):
        if "TARG_RA" in self[0].header:
            return self[0].header["TARG_RA"]
        else:
            return None

    @property
    def targ_dec(self):
        if "TARG_DEC" in self[0].header:
            return self[0].header["TARG_DEC"]
        else:
            return None

    @property
    def coord(self):
        if ("TARG_RA" in self[0].header) and ("TARG_DEC" in self[0].header):
            return SkyCoord(
                self[0].header["TARG_RA"],
                self[0].header["TARG_DEC"],
                unit="deg",
            )
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
        return convert_time(
            self[0].header["CORSTIME"], self[0].header["FINETIME"]
        )
        # time = (
        #     Time("2000-01-01 00:00:00", scale="tai")
        #     + timedelta(
        #         seconds=self[0].header["CORSTIME"],
        #         milliseconds=self[0].header["FINETIME"] / 1e6,
        #     )
        # ).utc
        # return time

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
    def read_time(self):
        return get_read_time(self[0].header)

    @property
    def exposure_time(self):
        return get_exposure_time(self[0].header)

    @property
    def end_time(self):
        return self.start_time + self.exposure_time

    # @property
    # def nframes(self):
    #     return self[1].header[f"NAXIS{self[1].header['NAXIS']}"]

    # @property
    # def ncoadds(self):
    #     if "FRMPCOAD" in self[0].header:
    #         return self[0].header["FRMPCOAD"]
    #     elif self[0].header["INSTRMNT"] == "VISDA":
    #         return 1
    #     if self[0].header["GRPSAVGD"] == 0:
    #         return 1
    #     else:
    #         return self[0].header["GRPS"]

    # @property
    # def end_time(self):
    #     return self.start_time + self.nframes * self.frame_time

    # @property
    # def time(self):
    #     dt = timedelta(seconds=self.frame_time.to(u.second).value)
    #     return self.start_time + (np.arange(self.nframes) * dt)

    @property
    def astrometry(self):
        if "astrometry" in self:
            if "JD" in self["astrometry"].data.columns.names:
                jd, ra, dec, rot = np.asarray(
                    [
                        self["astrometry"].data[c]
                        for c in [
                            "JD",
                            "RightAscension",
                            "Declination",
                            "Rotation",
                        ]
                    ]
                )
                jd = Time(jd, format="jd")
                return jd, ra, dec, rot
            else:
                ra, dec, rot = np.asarray(
                    [
                        self["astrometry"].data[c]
                        for c in ["RightAscension", "Declination", "Rotation"]
                    ]
                )
                et = self["TEMP_TIME"].data["ExposureStartTime_us"]
                et = et[: len(ra)]
                if et[-1] > 2**60:
                    et = et[:-1]
                    ra, dec, rot = ra[:-1], dec[:-1], rot[:-1]

                et = (
                    convert_time(
                        self[0].header["CORSTIME"], self[0].header["FINETIME"]
                    )
                    + et * u.ms
                )
                return et, ra, dec, rot
        else:
            raise ValueError("No `ASTROMETRY` extension.")

    def get_position_data(self):
        et, ra, dec, rot = self.astrometry
        df = []
        count, missing = [], []
        for t, dt in zip(self.time, self.exptime):
            j = (et >= (t)) & (et < (t + dt))
            k = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(rot)
            k &= (ra != 0) & (dec != 0) & (rot != 0)
            count.append(len(k[j]))
            missing.append((~k[j]).sum())
            k &= j
            if not k.any():
                df.append(
                    pd.DataFrame(
                        np.asarray([np.nan] * 6)[None, :],
                        columns=[
                            "avg_ra",
                            "avg_dec",
                            "avg_rot",
                            "err_ra",
                            "err_dec",
                            "err_rot",
                        ],
                    )
                )
                continue
            avg_ra, avg_dec, avg_rot = (
                np.nanmean(ra[k]),
                np.nanmean(dec[k]),
                np.nanmean(rot[k]),
            )
            err_ra, err_dec, err_rot = (
                np.nanmedian(np.abs(ra[k] - avg_ra)),
                np.nanmedian(np.abs(dec[k] - avg_dec)),
                np.nanmedian(np.abs(rot[k] - avg_rot)),
            )
            df.append(
                pd.DataFrame(
                    np.asarray(
                        [avg_ra, avg_dec, avg_rot, err_ra, err_dec, err_rot]
                    )[None, :],
                    columns=[
                        "avg_ra",
                        "avg_dec",
                        "avg_rot",
                        "err_ra",
                        "err_dec",
                        "err_rot",
                    ],
                )
            )
        df = pd.concat(df).reset_index(drop=True)
        t = self.time.jd
        df["jd"] = t
        df["exptime"] = self.exptime.value
        df["target_sep"] = np.hypot(
            df.avg_ra.values - self.coord.ra.value,
            df.avg_dec.values - self.coord.dec.value,
        )
        df["vitl_frames_count"] = count
        df["vitl_missing_frames"] = missing
        # if len(df) <= 3:
        #     df["stability"] = np.nan
        #     df["recall"] = np.nan
        #     df["stable"] = False
        # else:
        #     df["stability"] = (
        #         np.hypot(
        #             # np.gradient(df.err_ra.values, t), np.gradient(df.err_dec.values, t)
        #             df.err_ra.values,
        #             df.err_dec.values,
        #         )
        #         * 3600
        #     )
        #     df.loc[df["count"] < 5, "stability"] = np.nan
        #     df["recall"] = (
        #         np.hypot(
        #             np.gradient(df.avg_ra.values - np.nanmedian(df.avg_ra.values), t),
        #             np.gradient(df.avg_dec.values - np.nanmedian(df.avg_dec.values), t),
        #         )
        #         * 3600
        #         / 86400
        #     )
        earth_angle = self.get_earth_angle()
        df["earth_angle"] = earth_angle.value
        df["earth_illumination"] = self.earth_illumination.value
        df["sun_angle"] = self.sun_angle.value
        df["visda_keepout"] = earth_angle.value > self.visda_keepout.value
        df["nirda_keepout"] = earth_angle.value > self.nirda_keepout.value
        return df

    def plot_astrometry(self, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots()
        df = self.get_position_data()
        ax.plot(
            df.jd.values / 1e3,
            (df.avg_ra.values - df.avg_ra.mean()) * 3600,
            label="RA",
        )
        ax.plot(
            df.jd.values / 1e3,
            (df.avg_dec.values - df.avg_dec.mean()) * 3600,
            label="Dec",
        )
        ax.legend()
        ax.set(
            title=f"{self[0].header['targ_id']} {self.start_time.isot}",
            xlabel="Time in Exposure [s]",
            ylabel="Position - Mean Position [arcsecond]",
        )
        return ax

    # @cached_property
    # def earth_angle(self):
    #     return ps.get_angle_to_body(self.time, "z", "earth")

    def _get_angle(self, body, vec=None):
        """Get angle from SC boresight to the body"""
        if vec is None:
            if ("ra" in self[0].header) & ("dec" in self[0].header):
                ra, dec = self[0].header["ra"], self[0].header["dec"]
                vec = psc.utils.radec_to_vec(ra, dec)
            elif ("targ_ra" in self[0].header) & (
                "targ_dec" in self[0].header
            ):
                ra, dec = self[0].header["targ_ra"], self[0].header["targ_dec"]
                vec = psc.utils.radec_to_vec(ra, dec)
        return ps.get_angle_to_body(
            self.time,
            direction="z",
            body=body,
            pointing_vecs=vec,
        )

    def get_earth_angle(self, vec=None):
        return self._get_angle("earth", vec=vec)

    @cached_property
    def sun_angle(self):
        return self._get_angle("sun")

    @cached_property
    def earth_illumination(self):
        return ps.get_earth_illumination(self.time)

    @cached_property
    def visda_keepout(self):
        earth_illum = self.earth_illumination
        keepout = np.ones(len(earth_illum)) * 75
        keepout[earth_illum.value < 90] = (
            2.5 * (90 - earth_illum[earth_illum.value < 90].value) + 105
        )
        keepout[keepout > 135] = 135
        return keepout * u.deg

    @cached_property
    def nirda_keepout(self):
        earth_illum = self.earth_illumination
        keepout = np.ones(len(earth_illum)) * 75
        keepout[earth_illum.value < 90] = (
            2.5 * (90 - earth_illum[earth_illum.value < 90].value) + 80.5
        )
        keepout[keepout > 135] = 135
        return keepout * u.deg

    @cached_property
    def catalog_params(self):
        with gaiaoffline.Gaia(
            tmass_crossmatch=True, photometry_output="magnitude"
        ) as gaia:
            df = gaia.conesearch(self.targ_ra, self.targ_dec, 0.1).reset_index(
                drop=True
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coords = SkyCoord(
                ra=df["ra"].values * u.deg,
                dec=df["dec"].values * u.deg,
                pm_ra_cosdec=df["pmra"].fillna(0).values * u.mas / u.year,
                pm_dec=df["pmdec"].fillna(0).values * u.mas / u.year,
                obstime=Time.strptime("2016", "%Y"),
                distance=Distance(
                    parallax=df["parallax"].fillna(0).values * u.mas,
                    allow_negative=True,
                ),
                radial_velocity=df["radial_velocity"].fillna(0).values
                * u.km
                / u.s,
            ).apply_space_motion(self.start_time)
        df = df.iloc[coords.separation(self.coord).argmin()]
        return df.to_dict()

    def _score_file(self):
        mid = np.asarray(
            np.unravel_index(
                np.argmin(
                    np.hypot(
                        self.column[None, :] - self[0].header["posx"],
                        self.row[:, None] - self[0].header["posy"],
                    )
                ),
                (self.row.shape[0], self.column.shape[0]),
            )
        ).astype(float)
        mid[0] -= self.row.shape[0] / 2
        mid[1] -= self.column.shape[0] / 2

        useable = (
            np.isfinite(self["VECTORS"].data["avg_ra"])
            & (
                self["VECTORS"].data["jd"]
                > (self["VECTORS"].data["jd"][0] + (5 / (24 * 60)))
            )
            & self["VECTORS"].data["visda_keepout"]
            & self["VECTORS"].data["nirda_keepout"]
            & (self["VECTORS"].data["target_sep"] < (10 / 3600))
            & (
                np.abs(
                    (
                        self["VECTORS"].data["target_sep"]
                        - np.nanmedian(self["VECTORS"].data["target_sep"])
                    )
                )
                < (10 / 3600)
            )
        )
        k = np.isfinite(self["VECTORS"].data["avg_ra"])
        cards = [
            ("SRT_DATE", self.start_time.isot, "File Start Date"),
            ("END_DATE", self.end_time.isot, "File End Date"),
            (
                "FILETIME",
                (self.end_time - self.start_time).to(u.minute).value,
                "Time the file is observed for in minutes",
            ),
            (
                "VITLTIME",
                (
                    self["VECTORS"].data["jd"][k][-1]
                    - self["VECTORS"].data["jd"][k][0]
                )
                * (24 * 60),
                "Time VITL was on in minutes",
            ),
            (
                "ONTARG",
                100
                * (self["VECTORS"].data["target_sep"] < (10 / 3600)).sum()
                / self["VECTORS"].header["NAXIS2"],
                "Percentage On Target",
            ),
            (
                "NKEEPOUT",
                100
                * (
                    self["VECTORS"].data["nirda_keepout"].sum()
                    / self["VECTORS"].header["NAXIS2"]
                ),
                "Percentage of time NIRDA keepout obeyed",
            ),
            (
                "VKEEPOUT",
                100
                * (
                    self["VECTORS"].data["visda_keepout"].sum()
                    / self["VECTORS"].header["NAXIS2"]
                ),
                "Percentage of time VISDA keepout obeyed",
            ),
            (
                "VITLMISS",
                self["VECTORS"].data["vitl_missing_frames"].sum()
                / (self["VECTORS"].data["vitl_frames_count"]).sum(),
                "N times VITL return missing data",
            ),
            (
                "APCOMP1",
                100
                * self.aperture.sum()
                / (self["APERTURE"].data & 2 == 2).sum(),
                "Aperture Completeness Metric 1",
            ),
            ("TARGCENT", np.hypot(*mid) < 8, "Target Centered"),
            (
                "TARGCOMP",
                (
                    100
                    * self.aperture.sum()
                    / (self["APERTURE"].data & 2 == 2).sum()
                )
                > 0.7,
                "Target Complete",
            ),
            (
                "TARGTIME",
                np.median(self["VECTORS"].data["exptime"]) / 60 * useable.sum()
                > 10,
                "Usable Target Time is over 10 minutes",
            ),
        ]
        if "TCLDTIP2" in self[0].header:
            cards.append(
                (
                    "DETTEMP",
                    self[0].header["TCLDTIP2"] < 140,
                    "Detector temperature less than 140K",
                )
            )
        self[0].header.extend(cards)
