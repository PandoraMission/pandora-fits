import hashlib
import random
import string
from typing import List, Union

import numpy as np
import openpyxl
import pandas as pd

BITPIX_DICT = {
    8: (">u1", "Unsigned 8-bit integer, big-endian"),
    16: (">i2", "Signed 16-bit integer, big-endian"),
    32: (">i4", "Signed 32-bit integer, big-endian"),
    -32: (">f4", "32-bit floating-point (float32), big-endian"),
    -64: (">f8", "64-bit floating-point (float64), big-endian"),
}


# def strip_obs_count(targ_id):
#     """The SOC want to include a zeropadded integer at the end of all target names.
#     In case they don't do it for actually all target names, we have this strip function."""
#     if (targ_id[-4] == ".") & (targ_id[-3:].isnumeric()):
#         count = int(targ_id.split(".")[-1])
#     else:
#         count = 0
#     return targ_id, count


def get_dpc_hashkey(targ_id, ra, dec):
    key = (
        targ_id,
        np.round(np.nan_to_num(ra) if ra is not None else 0.0, 1),
        np.round(np.nan_to_num(dec) if dec is not None else 0.0, 1),
    )
    return hashlib.sha1(repr(key).encode()).hexdigest()[:8]


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


def generate_random_bintable_values(
    format_code: str, nrows: int
) -> Union[np.ndarray, list[str]]:
    """
    Generate random values suitable for astropy.io.fits BinTableHDU columns.

    Supported TFORM-style format codes:
      - 'D', 'nD'  : float64 scalar or fixed-length vector per row
      - 'K', 'nK'  : int64 scalar or fixed-length vector per row
      - 'wA'       : fixed-length ASCII string width w (e.g. '19A')

    Returns:
      - np.ndarray for numeric columns
      - list[str]  for 'A' columns (each exactly width characters)
    """
    fmt = format_code.strip().upper()

    # String: e.g. "19A"
    if fmt.endswith("A"):
        width_str = fmt[:-1]
        if not width_str.isdigit():
            raise ValueError(
                f"Unsupported string format: {format_code!r} (expected like '19A')"
            )
        width = int(width_str)

        alphabet = string.ascii_letters + string.digits + " _-"
        return ["".join(random.choices(alphabet, k=width)) for _ in range(nrows)]

    # Numeric: allow optional repeat count like "3D" or "10K"
    # If no leading digits, repeat=1.
    i = 0
    while i < len(fmt) and fmt[i].isdigit():
        i += 1
    repeat = int(fmt[:i]) if i > 0 else 1
    code = fmt[i:]

    if code == "D":
        # float64
        arr = np.random.uniform(-1e10, 1e10, size=(nrows, repeat)).astype(np.float64)
        return arr[:, 0] if repeat == 1 else arr

    if code == "K":
        # int64 (use a safer subset of full range to avoid edge-case overflows elsewhere)
        lo = np.iinfo(np.int64).min // 1000
        hi = np.iinfo(np.int64).max // 1000
        arr = np.random.randint(lo, hi, size=(nrows, repeat), dtype=np.int64)
        return arr[:, 0] if repeat == 1 else arr

    raise ValueError(f"Unsupported BinTableHDU format code: {format_code!r}")


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
                if row[0] not in ["", None, np.nan]
            ]
        )

    df = pd.DataFrame(rows)
    df.columns = df.loc[0]
    df = df[1:].reset_index(drop=True)
    df = df[~df.apply(lambda row: row.isnull().all(), axis=1)].reset_index(drop=True)
    return df
