from .base import *
from .vanillaoptions import *

from . import basic_american_options
from . import binomial_tree_options
from . import heston_nandi_options
from . import monte_carlo_options


def docstring_from(source):
    """
    Decorator to be used to copy the docstring from the source class/class method
    to the target. Useful for class composition.
    """

    def wrapper(target):
        target.__doc__ = source.__doc__
        return target

    return wrapper
