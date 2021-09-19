import sys
import os
import numpy as _np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src/")


import energyderivatives as ed


def test_PlainVanillaPayoff():

    S = 100
    K = 100
    t = 1 / 12
    sigma = 0.4
    r = 0.10
    b = 0.1

    dt = 1 / 360

    eps = _np.genfromtxt(
        "./pytest/sobol_path_test.csv", delimiter=","
    )  # load sobol paths from R since python version is slighly different in third path

    path = ed.monte_carlo_options.WienerPath(eps, sigma, dt, b)
    print(path.generate_path())
    payoff = ed.monte_carlo_options.PlainVanillaPayoff(
        path=path, S=S, K=K, t=t, sigma=sigma, r=r, b=b
    )

    # test two call options, elements 1 and 10 from fOptions
    # using weiner paths generated from sobol innovations
    assert _np.allclose(
        payoff.call()[[0, 9]], [37.11255, 37.83272]
    ), "PlainVanillaPayoff no matching R's fOptions."

    # test two put options, elements 1 and 10 from fOptions
    # using weiner paths generated from sobol innovations
    assert _np.allclose(
        payoff.put()[[0, 9]], [0, 0]
    ), "PlainVanillaPayoff no matching R's fOptions."

    # make K bigger so that the option value is > 0
    payoff = ed.monte_carlo_options.PlainVanillaPayoff(
        path=path, S=S, K=140, t=t, sigma=sigma, r=r, b=b
    )

    assert _np.allclose(
        payoff.put()[[0, 9]], [2.555497, 1.835328]
    ), "PlainVanillaPayoff no matching R's fOptions."


def test_ArimeticAsianPayoff():
    S = 100
    K = 100
    t = 1 / 12
    sigma = 0.4
    r = 0.10
    b = 0.1

    dt = 1 / 360

    eps = _np.genfromtxt(
        "./pytest/sobol_path_test.csv", delimiter=","
    )  # load sobol paths from R since python version is slighly different in third path

    path = ed.monte_carlo_options.WienerPath(eps, sigma, dt, b)

    payoff = ed.monte_carlo_options.ArithmeticAsianPayoff(
        path=path, S=S, K=K, t=t, sigma=sigma, r=r, b=b
    )

    # test two call options, elements 1 and 10 from fOptions
    # using weiner paths generated from sobol innovations
    assert _np.allclose(
        payoff.call()[[0, 9]], [18.19441, 20.43197]
    ), "ArithmeticAsianPayoff not matching R's fOptions."

    # test two put options, elements 1 and 10 from fOptions
    # using weiner paths generated from sobol innovations
    assert _np.allclose(
        payoff.put()[[0, 9]], [0, 0]
    ), "ArithmeticAsianPayoff not matching R's fOptions."

    # make K bigger so that the option value is > 0
    payoff = ed.monte_carlo_options.ArithmeticAsianPayoff(
        path=path, S=S, K=140, t=t, sigma=sigma, r=r, b=b
    )

    assert _np.allclose(
        payoff.put()[[0, 9]], [21.47364, 19.23608]
    ), "ArithmeticAsianPayoff not matching R's fOptions."


if __name__ == "__main__":
    test_PlainVanillaPayoff()
