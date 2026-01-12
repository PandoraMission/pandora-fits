import tempfile

import numpy as np
import pandas as pd
from astropy.io import fits

from . import FORMATSDIR
from .nirda import NIRDALevel0HDUList
from .visda import VISDAFFILevel0HDUList, VISDALevel0HDUList


def hdulist_to_dataframes(hdulist):
    fixed_keys = [
        "SIMPLE",
        "BITPIX",
        "XTENSION",
        "NAXIS",
        "EXTEND",
        "EXTNAME",
        "TELESCOP",
        "CAMERAID",
        "INSTRMNT",
        "TFIELDS",
        "BSCALE",
        "BZERO",
    ]
    fixed_keys = np.hstack(
        [
            fixed_keys,
            *[
                [
                    tabkey + f"{idx}"
                    for tabkey in ["TTYPE", "TFORM", "TBCOL"]
                    for idx in range(40)
                ]
            ],
        ]
    )
    if isinstance(hdulist, str):
        hdulist = fits.open(hdulist)
    # We do this because sometimes when we write these files they seem to gain extra keywords
    with tempfile.NamedTemporaryFile(suffix=".fits") as tmp:
        filename = tmp.name
        hdulist.writeto(filename, overwrite=True, checksum=True)
        hdulist = fits.open(filename)

    dfs = pd.DataFrame(
        [
            np.asarray([hdu.header["EXTNAME"] for hdu in hdulist]),
            np.asarray([type(hdu).__name__ for hdu in hdulist]),
            np.zeros(len(hdulist), bool),
        ]
    ).T
    dfs.columns = ["Extension", "Type", "Optional"]
    dfs = [
        dfs,
        *[
            pd.DataFrame(
                np.asarray(
                    [
                        np.asarray(card)
                        for card in hdu.header.cards
                        if card[0] is not None
                    ]
                ),
                columns=["Name", "Example Value", "Comment"],
            )
            for hdu in hdulist
        ],
    ]
    for df in dfs[1:]:
        df["Fixed Value"] = None

    for idx, df in enumerate(dfs):
        if idx == 0:
            continue
        df = df.set_index("Name")
        for key in fixed_keys:
            if key in df.index:
                df.loc[key, "Fixed Value"] = df.loc[key, "Example Value"]
        dfs[idx] = df.reset_index()
    return dfs


def hdulist_to_excel(hdulist, filename="output.xlsx"):
    sheet_names = ["Primary", *[f"Ext{idx}" for idx in np.arange(1, len(hdulist))]]
    dfs = hdulist_to_dataframes(hdulist)
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        dfs[0].to_excel(writer, sheet_name="Structure", index=False)
        for name, df in zip(sheet_names, dfs[1:]):
            df[["Name", "Fixed Value", "Example Value", "Comment"]].to_excel(
                writer, sheet_name=name, index=False
            )


def update_VISDAFFI_format(filepath, level=2):
    hdulist0 = fits.open(filepath)
    hdulist_to_excel(hdulist0, f"{FORMATSDIR}visda/level0-ffi_visda.xlsx")
    if level == 0:
        return
    hdulist1 = VISDAFFILevel0HDUList(filepath).to_level1(
        upcast=False  # , targ_ra=targ_ra, targ_dec=targ_dec, targ_rll=targ_rll
    )
    # add_pointing_params(hdulist1)
    hdulist_to_excel(hdulist1, f"{FORMATSDIR}visda/level1-ffi_visda.xlsx")
    if level == 1:
        return
    hdulist2 = (
        VISDAFFILevel0HDUList(filepath)
        .to_level1()  # targ_ra=targ_ra, targ_dec=targ_dec, targ_rll=targ_rll)
        .to_level2(upcast=False)
    )
    # add_pointing_params(hdulist2)
    hdulist_to_excel(hdulist2, f"{FORMATSDIR}visda/level2-ffi_visda.xlsx")
    if level == 2:
        return


def update_VISDA_format(filepath, level=2):
    hdulist0 = fits.open(filepath)
    hdulist_to_excel(hdulist0, f"{FORMATSDIR}visda/level0_visda.xlsx")
    if level == 0:
        return
    hdulist1 = VISDALevel0HDUList(filepath).to_level1(upcast=False)
    # add_pointing_params(hdulist1)
    hdulist_to_excel(hdulist1, f"{FORMATSDIR}visda/level1_visda.xlsx")
    if level == 1:
        return
    hdulist2 = VISDALevel0HDUList(filepath).to_level1().to_level2(upcast=False)
    # add_pointing_params(hdulist2)
    hdulist_to_excel(hdulist2, f"{FORMATSDIR}visda/level2_visda.xlsx")
    if level == 2:
        return


def update_NIRDA_format(filepath, level=2):
    hdulist0 = fits.open(filepath)
    hdulist_to_excel(hdulist0, f"{FORMATSDIR}nirda/level0_nirda.xlsx")
    if level == 0:
        return
    hdulist1 = NIRDALevel0HDUList(filepath).to_level1(
        upcast=False  # ), targ_ra=targ_ra, targ_dec=targ_dec, targ_rll=targ_rll
    )
    # add_pointing_params(hdulist1)
    hdulist_to_excel(hdulist1, f"{FORMATSDIR}nirda/level1_nirda.xlsx")
    if level == 1:
        return
    hdulist2 = (
        NIRDALevel0HDUList(filepath)
        .to_level1()
        .to_level2(
            upcast=False  # , targ_ra=targ_ra, targ_dec=targ_dec, targ_rll=targ_rll
        )
    )
    # add_pointing_params(hdulist2)
    hdulist_to_excel(hdulist2, f"{FORMATSDIR}nirda/level2_nirda.xlsx")
    if level == 2:
        return


def update_formats(example_file_path, level=2):
    if "VisImg" in example_file_path:
        update_VISDAFFI_format(example_file_path, level=level)
    if "VisSci" in example_file_path:
        update_VISDA_format(example_file_path, level=level)
    if "InfImg" in example_file_path:
        update_NIRDA_format(example_file_path, level=level)
