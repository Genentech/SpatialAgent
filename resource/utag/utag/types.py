#!/usr/bin/env python

"""
Specific data types used for type annotations in the package.
"""

from __future__ import annotations
import os
import typing as tp
import pathlib


import numpy
import pandas
import anndata
import networkx
import matplotlib
from matplotlib.figure import Figure as _Figure


__all__ = [
    "Array",
    "Graph",
    "DataFrame",
    "Figure",
    "Axis",
    "Path",
    "AnnData",
]


class Path(pathlib.Path):
    """
    A pathlib.Path child class that allows concatenation with strings
    by overloading the addition operator.

    In addition, it implements the ``startswith`` and ``endswith`` methods
    just like in the base :obj:`str` type.

    The ``replace_`` implementation is meant to be an implementation closer
    to the :obj:`str` type.

    Iterating over a directory with ``iterdir`` that does not exists
    will return an empty iterator instead of throwing an error.

    Creating a directory with ``mkdir`` allows existing directory and
    creates parents by default.
    """

    # Python 3.12+ no longer requires _flavour for Path subclassing
    # The old code used pathlib._posix_flavour/_windows_flavour which were removed

    def __add__(self, string: str) -> Path:
        return Path(str(self) + string)

    def startswith(self, string: str) -> bool:
        return str(self).startswith(string)

    def endswith(self, string: str) -> bool:
        return str(self).endswith(string)

    def replace_(self, patt: str, repl: str) -> Path:
        return Path(str(self).replace(patt, repl))

    def iterdir(self) -> tp.Generator:
        if self.exists():
            yield from [Path(x) for x in pathlib.Path(str(self)).iterdir()]
        yield from []

    def mkdir(self, mode=0o777, parents: bool = True, exist_ok: bool = True) -> Path:
        super().mkdir(mode=mode, parents=parents, exist_ok=exist_ok)
        return self


Array = tp.Union[numpy.ndarray]
Graph = tp.Union[networkx.Graph]

DataFrame = tp.Union[pandas.DataFrame]
AnnData = tp.Union[anndata.AnnData]

Figure = tp.Union[_Figure]
Axis = tp.Union[matplotlib.axis.Axis]
