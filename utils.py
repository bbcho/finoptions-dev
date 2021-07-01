import pandas as _pd
import numpy as _np


def check_iter(it):
    # try:
    #     _ = (e for e in it)
    # except TypeError:
    #     print(it, "is not iterable")

    return hasattr(it, "__iter__")

