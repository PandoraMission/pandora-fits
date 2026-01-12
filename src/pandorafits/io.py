"""Deal with opening files nicely"""

from __future__ import annotations

import functools
import inspect
from typing import Callable, List, Tuple, Type

from astropy.io import fits

Predicate = Callable[[fits.HDUList], bool]
_REGISTRY: List[Tuple[Predicate, Type[fits.HDUList]]] = []

__all__ = ["open"]


def register_hdulist(predicate: Predicate):
    """Decorator to register an HDUList subclass for auto-detection."""

    def deco(cls: Type[fits.HDUList]):
        _REGISTRY.append((predicate, cls))
        return cls

    return deco


@functools.wraps(fits.open)
def open(*args, **kwargs):
    """
    Like astropy.io.fits.open, but returns an appropriate HDUList subclass
    when one matches the opened file.
    """
    hdul = fits.open(*args, **kwargs)

    for pred, cls in _REGISTRY:
        try:
            if pred(hdul):
                # "cast" in-place so behavior (file handle, lazy loading, etc.) stays identical
                hdul.__class__ = cls
                return hdul
        except Exception:
            # Detection should never break opening; just skip failed detectors.
            continue

    return hdul


# Optional: make your wrapper show the same signature in IDEs/help()
try:
    open.__signature__ = inspect.signature(fits.open)
except Exception:
    pass
