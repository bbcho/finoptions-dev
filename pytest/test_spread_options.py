import sys
import os
import numpy as np


sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src/")

import finoptions as fo

S1 = 122
S2 = 120
K = 3
r = 0.1
b1 = 0
b2 = 0
n = 100

testsBi = [
    dict(
        params=dict(sigma1=0.2, sigma2=0.2, rho=-0.5, t=0.1, otype="european"),
        vals=dict(call=4.7554),
    ),
    dict(
        params=dict(sigma1=0.2, sigma2=0.2, rho=0, t=0.1, otype="european"),
        vals=dict(call=3.8008),
    ),
    dict(
        params=dict(sigma1=0.2, sigma2=0.2, rho=0.5, t=0.1, otype="european"),
        vals=dict(call=2.5551),
    ),
    dict(
        params=dict(sigma1=0.2, sigma2=0.2, rho=-0.5, t=0.5, otype="european"),
        vals=dict(call=10.7566),
    ),
    dict(
        params=dict(sigma1=0.2, sigma2=0.2, rho=0, t=0.5, otype="european"),
        vals=dict(call=8.7080),
    ),
    dict(
        params=dict(sigma1=0.2, sigma2=0.2, rho=0.5, t=0.5, otype="european"),
        vals=dict(call=6.0286),
    ),
    ##############
    dict(
        params=dict(sigma1=0.25, sigma2=0.2, rho=-0.5, t=0.1, otype="european"),
        vals=dict(call=5.4297),
    ),
    dict(
        params=dict(sigma1=0.25, sigma2=0.2, rho=0, t=0.1, otype="european"),
        vals=dict(call=4.3732),
    ),
    dict(
        params=dict(sigma1=0.25, sigma2=0.2, rho=0.5, t=0.1, otype="european"),
        vals=dict(call=3.0098),
    ),
    dict(
        params=dict(sigma1=0.25, sigma2=0.2, rho=-0.5, t=0.5, otype="european"),
        vals=dict(call=12.2031),
    ),
    dict(
        params=dict(sigma1=0.25, sigma2=0.2, rho=0, t=0.5, otype="european"),
        vals=dict(call=9.9377),
    ),
    dict(
        params=dict(sigma1=0.25, sigma2=0.2, rho=0.5, t=0.5, otype="european"),
        vals=dict(call=7.0097),
    ),
    ################
    ################
    dict(
        params=dict(sigma1=0.2, sigma2=0.2, rho=-0.5, t=0.1, otype="american"),
        vals=dict(call=4.7630),
    ),
    dict(
        params=dict(sigma1=0.2, sigma2=0.2, rho=0, t=0.1, otype="american"),
        vals=dict(call=3.8067),
    ),
    dict(
        params=dict(sigma1=0.2, sigma2=0.2, rho=0.5, t=0.1, otype="american"),
        vals=dict(call=2.5590),
    ),
    dict(
        params=dict(sigma1=0.2, sigma2=0.2, rho=-0.5, t=0.5, otype="american"),
        vals=dict(call=10.8754),
    ),
    dict(
        params=dict(sigma1=0.2, sigma2=0.2, rho=0, t=0.5, otype="american"),
        vals=dict(call=8.8029),
    ),
    dict(
        params=dict(sigma1=0.2, sigma2=0.2, rho=0.5, t=0.5, otype="american"),
        vals=dict(call=6.0939),
    ),
    ##############
    dict(
        params=dict(sigma1=0.25, sigma2=0.2, rho=-0.5, t=0.1, otype="american"),
        vals=dict(call=5.4385),
    ),
    dict(
        params=dict(sigma1=0.25, sigma2=0.2, rho=0, t=0.1, otype="american"),
        vals=dict(call=4.3802),
    ),
    dict(
        params=dict(sigma1=0.25, sigma2=0.2, rho=0.5, t=0.1, otype="american"),
        vals=dict(call=3.0145),
    ),
    dict(
        params=dict(sigma1=0.25, sigma2=0.2, rho=-0.5, t=0.5, otype="american"),
        vals=dict(call=12.3383),
    ),
    dict(
        params=dict(sigma1=0.25, sigma2=0.2, rho=0, t=0.5, otype="american"),
        vals=dict(call=10.0468),
    ),
    dict(
        params=dict(sigma1=0.25, sigma2=0.2, rho=0.5, t=0.5, otype="american"),
        vals=dict(call=7.0858),
    ),
]


# sigma1 = 0.25
# sigma2 = 0.2
# rho = -0.5
# t = 0.1
# otype = "european"


def test_binomial_spread():

    for i, test in enumerate(testsBi):

        opt = fo.spread_options.BionomialSpreadOption(
            S1=S1,
            S2=S2,
            K=K,
            r=r,
            b1=b1,
            b2=b2,
            n=n,
            **test["params"],
        )
        testval = opt.call()
        assert np.allclose(
            round(testval, 4), test["vals"]["call"]
        ), f"Bionomial Spread Option call test {str(i)} failed. Params = {test['params']}. Test = {test['vals']['call']}. Return {testval}"

        if test["params"]["otype"] == "european":
            # uses put-call parity relationship for euro options to test puts. Does not apply to american options
            # fmt: off
            val = opt.call() - (opt._S1*np.exp((opt._b1-opt._r)*opt._t) - opt._S2*np.exp((opt._b2-opt._r)*opt._t)) + opt._K*np.exp(-opt._r*opt._t)
            testval = opt.put()
            call = test["vals"]["call"]
            assert np.allclose(
                round(testval, 3), round(val, 3)
            ), f"Bionomial Spread Option put test {str(i)}, {call} failed. Params = {test['params']}. Test = {val}. Return {testval}"
            # fmt: on


testsApprox = [
    dict(
        params=dict(sigma1=0.2, sigma2=0.2, rho=-0.5, t=0.1),
        vals=dict(call=4.7530),
    ),
    dict(
        params=dict(sigma1=0.2, sigma2=0.2, rho=0, t=0.1),
        vals=dict(call=3.7970),
    ),
    dict(
        params=dict(sigma1=0.2, sigma2=0.2, rho=0.5, t=0.1),
        vals=dict(call=2.5537),
    ),
    dict(
        params=dict(sigma1=0.2, sigma2=0.2, rho=-0.5, t=0.5),
        vals=dict(call=10.7517),
    ),
    dict(
        params=dict(sigma1=0.2, sigma2=0.2, rho=0, t=0.5),
        vals=dict(call=8.7020),
    ),
    dict(
        params=dict(sigma1=0.2, sigma2=0.2, rho=0.5, t=0.5),
        vals=dict(call=6.0257),
    ),
    ##############
    dict(
        params=dict(sigma1=0.25, sigma2=0.2, rho=-0.5, t=0.1),
        vals=dict(call=5.4275),
    ),
    dict(
        params=dict(sigma1=0.25, sigma2=0.2, rho=0, t=0.1),
        vals=dict(call=4.3712),
    ),
    dict(
        params=dict(sigma1=0.25, sigma2=0.2, rho=0.5, t=0.1),
        vals=dict(call=3.0086),
    ),
    dict(
        params=dict(sigma1=0.25, sigma2=0.2, rho=-0.5, t=0.5),
        vals=dict(call=12.1941),
    ),
    dict(
        params=dict(sigma1=0.25, sigma2=0.2, rho=0, t=0.5),
        vals=dict(call=9.9340),
    ),
    dict(
        params=dict(sigma1=0.25, sigma2=0.2, rho=0.5, t=0.5),
        vals=dict(call=7.0067),
    ),
]


def test_approx_spread():

    for i, test in enumerate(testsApprox):

        opt = fo.spread_options.SpreadApproxOption(
            S1=S1,
            S2=S2,
            K=K,
            r=r,
            b1=b1,
            b2=b2,
            **test["params"],
        )
        testval = opt.call()
        assert np.allclose(
            round(testval, 4), test["vals"]["call"]
        ), f"Approximate Spread Option call test {str(i)} failed. Params = {test['params']}. Test = {test['vals']['call']}. Return {testval}"

        testval = opt.put()

        # fmt: off
        val = opt.call() - (opt._S1*np.exp(opt._b1-opt._r*opt._t) - opt._S2*np.exp(opt._b2-opt._r*opt._t)) + opt._K*np.exp(-opt._r*opt._t)
        assert np.allclose(
            round(testval, 4), val
        ), f"Approximate Spread Option put test {str(i)} failed. Params = {test['params']}. Test = {test['vals']['call']}. Return {testval}"
        # fmt: on


if __name__ == "__main__":
    test_binomial_spread()
