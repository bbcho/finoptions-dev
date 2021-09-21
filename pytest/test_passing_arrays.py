import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src/")

import finoptions as fo


def test_vanilla_array():
    #fmt: off
    opt_test = [
        (fo, "GBSOption", {"S": 10.0, "t": 1.0, "r": 0.02, "b": 0.01, "sigma": 0.2}, {'vol':True}),
        (fo, "BlackScholesOption", {"S": 10.0, "t": 1.0, "r": 0.02, "b": 0.01, "sigma": 0.2}, {'vol':True}),
        (fo, "Black76Option", {"FT": 10.0, "t": 1.0, "r": 0.02, "sigma": 0.2}, {'vol':True}),
        (fo, "MiltersenSchwartzOption", dict(Pt=np.exp(-0.05 / 4), FT=10, t=1 / 4, T=1 / 2, sigmaS=0.2660, sigmaE=0.2490, sigmaF=0.0096, 
                    rhoSE=0.805, rhoSF=0.0805, rhoEF=0.1243, KappaE=1.045,KappaF=0.200), {'vol':False})
    ]

    K = np.arange(8, 12)

    for test in opt_test:
        # create object with array of K
        opt_array = getattr(test[0], test[1])(K=K, **test[2])

        # create empty list to hold object where elements of K are pass individually
        opt = []
        for k_ind in K:
            # create object for each element of K
            opt.append(getattr(test[0], test[1])(K=k_ind, **test[2]))

        for i, _ in enumerate(opt):
            assert np.allclose(
                opt_array.call()[i], opt[i].call()
            ), f"{opt[i].__name__} failed call() test for passing arrays for K = {K[i]}"

        for i, _ in enumerate(opt):
            assert np.allclose(
                opt_array.put()[i], opt[i].put()
            ), f"{opt[i].__name__} failed put() test for passing arrays for K = {K[i]}"

        if test[3]['vol'] == True:
            for i, _ in enumerate(opt):
                assert np.allclose(
                    opt_array.volatility(K - 5)[i], opt[i].volatility(K[i] - 5)
                ), f"{opt[i].__name__} failed volatility() test for passing arrays for K = {K[i]}"

        # test greeks
        for gk in ["delta", "theta", "vega", "rho", "gamma", "lamb"]:
            for i, _ in enumerate(opt):
                if (gk != 'rho') & (opt[i].__name__ != "MiltersenSchwartzOption"):
                    print(opt[i].__name__, gk, getattr(opt_array, gk)()[i], getattr(opt[i], gk)())
                    assert np.allclose(
                        getattr(opt_array, gk)()[i], getattr(opt[i], gk)()
                    ), f"{opt[i].__name__} failed {gk}() test for passing arrays for K = {K[i]}"


if __name__ == "__main__":
    test_vanilla_array()
