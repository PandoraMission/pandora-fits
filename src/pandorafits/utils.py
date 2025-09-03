import numpy as np
import random
import string
from typing import List

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
