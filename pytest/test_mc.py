import sys
import os
import numpy as _np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src/")


import energyderivatives as ed


def test_monte_carlo():

    S = 100
    K = 100
    t = 1 / 12
    sigma = 0.4
    r = 0.10
    b = 0.1

    dt = 1 / 360

    path_length = 30
    mc_samples = 5000
    mc_loops = 500

    eps = _np.genfromtxt(
        "./pytest/sobol_path_test.csv", delimiter=","
    )  # load sobol paths from R since python version is slighly different in third path

    inno = ed.monte_carlo_options.NormalSobolInnovations
    path = ed.monte_carlo_options.WienerPath
    payoff = ed.monte_carlo_options.PlainVanillaPayoff

    mc = ed.monte_carlo_options.MonteCarloOption(
        mc_loops,
        path_length,
        mc_samples,
        dt,
        S,
        K,
        t,
        r,
        b,
        sigma,
        inno,
        path,
        payoff,
        # eps=eps,
        trace=False,
        antithetic=True,
    )

    opt = ed.GBSOption(S, K, t, r, b, sigma)

    assert _np.allclose(
        opt.call(), _np.mean(mc.call()), rtol=1e-3
    ), "Monte Carlo Plain Vanilla call failed"
    assert _np.allclose(
        opt.put(), _np.mean(mc.put()), rtol=1e-3
    ), "Monte Carlo Plain Vanilla put failed"


if __name__ == "__main__":

    test_monte_carlo()
