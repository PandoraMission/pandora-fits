import random
import string
from typing import List

import numpy as np
import openpyxl
import pandas as pd
from astropy.io import fits

BITPIX_DICT = {
    8: (">u1", "Unsigned 8-bit integer, big-endian"),
    16: (">i2", "Signed 16-bit integer, big-endian"),
    32: (">i4", "Signed 32-bit integer, big-endian"),
    -32: (">f4", "32-bit floating-point (float32), big-endian"),
    -64: (">f8", "64-bit floating-point (float64), big-endian"),
}


def generate_random_table_values(format_code: str, nvalues: int) -> List:
    if format_code.startswith("A"):
        # Character string
        width = int(format_code[1:])
        return [
            "".join(random.choices(string.ascii_letters + string.digits, k=width))
            for n in range(nvalues)
        ]

    elif format_code.startswith("I"):
        # Integer
        width = np.min([int(format_code[1:]), 10])
        return [
            str(random.randint(10 ** (width - 1), 10**width - 1)).rjust(width)
            for n in range(nvalues)
        ]

    elif format_code.startswith("F"):
        # Fixed floating point
        parts = format_code[1:].split(".")
        width = int(parts[0]) - 1
        decimal_places = int(parts[1])
        value = [
            round(
                random.uniform(
                    10 ** (width - decimal_places - 1),
                    10 ** (width - decimal_places) - 1,
                ),
                decimal_places,
            )
            for n in range(nvalues)
        ]
        return [f"{value[n]:>{width}.{decimal_places}f}" for n in range(nvalues)]

    elif format_code.startswith("E") or format_code.startswith("D"):
        # Exponential floating point
        parts = format_code[1:].split(".")
        width = int(parts[0])
        decimal_places = int(parts[1])
        value = random.uniform(
            1e-10, 1e10
        )  # Generating a random number with a large range
        if format_code.startswith("D"):
            return [
                f"{value:>{width}.{decimal_places}E}" for n in range(nvalues)
            ]  # Use 'E' to match exponential format with width
        else:
            return [
                f"{value:>{width}.{decimal_places}e}" for n in range(nvalues)
            ]  # 'e' is used for lowercase exponent notation

    else:
        raise ValueError("Unsupported format code")


def get_excel_sheet(fname, extno=0):
    wb = openpyxl.load_workbook(fname, data_only=False)
    ws = wb.worksheets[extno]

    rows = []
    for row in ws.iter_rows(values_only=False):
        rows.append(
            [
                (
                    cell.value
                    if cell.data_type != "n"
                    else cell.number_format and cell._value
                )
                for cell in row
            ]
        )

    df = pd.DataFrame(rows)
    df.columns = df.loc[0]
    df = df[1:].reset_index(drop=True)
    df = df[~df.apply(lambda row: row.isnull().all(), axis=1)].reset_index(drop=True)
    return df


def hdulist_to_excel(hdulist, filename="output.xlsx"):
    if isinstance(hdulist, str):
        hdulist = fits.open(hdulist)

    sheet_names = ["Primary", *[f"Ext{idx}" for idx in np.arange(1, len(hdulist))]]
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
                np.asarray([np.asarray(card) for card in hdu.header.cards]),
                columns=["Name", "Example Value", "Comment"],
            )
            for hdu in hdulist
        ],
    ]
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        dfs[0].to_excel(writer, sheet_name="Structure", index=False)
        for name, df in zip(sheet_names, dfs[1:]):
            df["Fixed Value"] = None
            df[["Name", "Fixed Value", "Example Value", "Comment"]].to_excel(
                writer, sheet_name=name, index=False
            )
