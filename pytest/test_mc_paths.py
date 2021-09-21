import sys
import os
import numpy as np
from matplotlib import figure

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src/")

import finoptions as fo


def test_WienerPath():
    eps = np.genfromtxt(
        "./pytest/sobol_path_test.csv", delimiter=","
    )  # load sobol paths from R since python version is slighly different in third path
    rpaths = np.genfromtxt(
        "./pytest/wiener_path_test.csv", delimiter=","
    )  # results to test against
    paths = fo.monte_carlo_options.WienerPath(eps, 0.4, 1 / 360, 0.1).generate_path()

    assert np.allclose(
        paths, rpaths
    ), "WienerPath results do not match fOptions from R."
