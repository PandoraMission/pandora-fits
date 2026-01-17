# Standard library
import functools

# Third-party
import numpy.typing as npt
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.units import Quantity

DOCSTRINGS = {
    "border": (
        bool,
        "Whether to include a 1 pixel border around insets within a panel in ROI mode.",
    ),
    "nROI": (int, "The number of regions of interest in the larger image"),
    "ROI_size": (
        tuple,
        "The size the regions of interest in (row, column) pixels. All ROIs must be the same size.",
    ),
    "ROI_corners": (
        list,
        "The origin (lower left) corner positon for each of the ROIs. Must have length nROIs. List of tuples.",
    ),
    "ROI_array": (
        npt.NDArray,
        "Array of ROIs for VisSci images. Arrays have shape (ntime, nROI, ROI_size, ROI_size) or (nROI, ROI_size, ROI_size) if there is no time dimension.",
    ),
    "ROI_cube": (
        npt.NDArray,
        "Cube of ROIs for VisSci images. Cubes have shape (ntime, n, n, ROI_size, ROI_size) or (n, n, ROI_size, ROI_size) if there is no time dimension. n is the ceil of the square root of nROI.",
    ),
    "ROI_panels": (
        npt.NDArray,
        "Panel of ROIs for VisSci images. Panels are 2D if there is no time dimension or 3D if there is a time dimension, where the first dimension is time. Panels contain all ROIs as insets.",
    ),
    "time": (Time, "A time as an astropy.time.Time object."),
    "target": (
        SkyCoord,
        "A target as an astropy.coordinates.SkyCoord object.",
    ),
    "roll": (
        Quantity,
        "The roll of the spacecraft as an astropy.units.Quantity with units of degrees.",
    ),
}


def extract_docstring_type(dtype, desc):
    if isinstance(dtype, tuple):
        dtype_str = " or ".join(
            [t._name if hasattr(t, "_name") else t.__name__][0]
            for t in dtype
            if t is not None
        )
        dtype_str += " or None" if None in dtype else ""
    elif isinstance(dtype, str):
        dtype_str = dtype
    else:
        dtype_str = dtype.__name__
    return dtype_str, desc


def clean_docstring(func, additional_docstring, indent_str, heading):
    existing_docstring = func.__doc__ or ""
    if heading in existing_docstring:
        func.__doc__ = (
            existing_docstring.split("---\n")[0]
            + "---\n"
            + additional_docstring
            + "---\n".join(existing_docstring.split("---\n")[1:])
        )
    else:
        func.__doc__ = (
            existing_docstring
            + f"\n\n{indent_str}{heading}\n{indent_str}----------\n"
            + additional_docstring
        )


# Decorator to add common parameters to docstring
def add_docstring(func=None, *, parameters=None, returns=None):
    def decorator(func, parameters=parameters, returns=returns):
        param_docstring, return_docstring = "", ""
        if func.__doc__:
            # Determine the current indentation level
            lines = func.__doc__.splitlines()
            if len(lines[0]) == 0:
                indent = len(lines[1]) - len(lines[1].lstrip())
            else:
                indent = len(lines[0]) - len(lines[0].lstrip())
        else:
            indent = 0
        indent_str = " " * indent
        if isinstance(parameters, str):
            parameters = [parameters]
        if isinstance(returns, str):
            returns = [returns]

        if parameters is not None:
            for name in parameters:
                if name in DOCSTRINGS:
                    dtype_str, desc = extract_docstring_type(*DOCSTRINGS[name])
                    param_docstring += f"{indent_str}{name}: {dtype_str}\n{indent_str}    {desc}\n"
            clean_docstring(func, param_docstring, indent_str, "Parameters")

        if returns is not None:
            for name in returns:
                if name in DOCSTRINGS:
                    dtype_str, desc = extract_docstring_type(*DOCSTRINGS[name])
                    return_docstring += f"{indent_str}{name}: {dtype_str}\n{indent_str}    {desc}\n"
            clean_docstring(func, return_docstring, indent_str, "Returns")
        return func

    return decorator


# Decorator to inherit docstring from base class
def inherit_docstring(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    if func.__doc__ is None:
        for base in func.__qualname__.split(".")[0].__bases__:
            base_func = getattr(base, func.__name__, None)
            if base_func and base_func.__doc__:
                func.__doc__ = base_func.__doc__
                break
    return wrapper
