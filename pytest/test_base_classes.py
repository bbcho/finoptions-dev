import sys
import os
import numpy as np


sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src/")

import finoptions as fo


# def test_Option():
#     b = fo.Option(0, 0, 0, 0, 0, 0)

#     assert isinstance(b, fo.Option), "Option object failed init"

#     assert isinstance(
#         b.get_params(), dict
#     ), "Option.get_params failed to return dictionary of values"


def test_GBSOption():

    opt = fo.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)

    assert (
        round(opt.call(), 6) == 2.061847
    ), "GBSOption call price does not match fOptions. GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1).call() should equal 2.061847"

    assert (
        round(opt.put(), 8) == 0.00293811
    ), "GBSOption put price does not match fOptions. GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1).put() should equal 0.00293811"

    # test greeks, rounded to 6 decimal points
    greeks = {
        True: {
            "delta": 0.981513,
            "theta": -0.068503,
            "vega": 0.231779,
            "rho": 7.753283,
            "lambda": 4.760358,
            "gamma": 0.023178,
            "CofC": 9.81513,
        },
        False: {
            "delta": -0.008537,
            "theta": -0.010677,
            "vega": 0.231779,
            "rho": -0.088307,
            "lambda": -29.055576,
            "gamma": 0.023178,
            "CofC": -0.085368,
        },
    }

    for op in greeks.keys():
        print(opt)
        test_greeks = opt.greeks(call=op)
        if op == True:
            type = "call"
        else:
            type = "put"
        for cp in greeks[op].keys():

            my_val = round(test_greeks[cp], 6)
            control_val = round(greeks[op][cp], 6)
            assert (
                my_val == control_val
            ), f"GBSOption {cp} calculation for a {type} option does not match fOptions. GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1) should result in {cp} of {control_val}"

    # test implied volatility method
    vol = fo.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01)
    assert (
        round(vol.volatility(2.5, call=True), 6) == 0.342241
    ), "GBSOption implied volatility calculation does not match fOptions for a call option. GBSOption(10.0, 8.0, 1.0, 0.02, 0.01).volatility(3) should equal 0.342241"
    assert (
        round(vol.volatility(2.5, call=False), 6) == 1.016087
    ), "GBSOption implied volatility calculation does not match fOptions for a call option. GBSOption(10.0, 8.0, 1.0, 0.02, 0.01).volatility(3) should equal 1.016087"

    assert isinstance(
        opt.summary(printer=False), str
    ), "GBSOption.summary() failed to produce string."


def test_BlackScholesOption():

    opt = fo.BlackScholesOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)

    assert (
        round(opt.call(), 6) == 2.061847
    ), "BlackScholesOption call price does not match fOptions. BlackScholesOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1).call() should equal 2.061847"


def test_Black76sOption():

    opt = fo.Black76Option(10.0, 8.0, 1.0, 0.02, 0.1)

    assert (
        round(opt.call(), 6) == 1.96431
    ), "Black76Option call price does not match fOptions. Black76Option(10.0, 8.0, 1.0, 0.02, 0.1).call() should equal 1.96431"


def test_MiltersenSchwartzOption():
    opt = fo.MiltersenSchwartzOption(
        Pt=np.exp(-0.05 / 4),
        FT=95,
        K=80,
        t=1 / 4,
        T=1 / 2,
        sigmaS=0.2660,
        sigmaE=0.2490,
        sigmaF=0.0096,
        rhoSE=0.805,
        rhoSF=0.0805,
        rhoEF=0.1243,
        KappaE=1.045,
        KappaF=0.200,
    )

    assert (
        round(opt.call(), 5) == 15.00468
    ), "MiltersenSchwartzOption call price does not match fOptions. MiltersenSchwartzOption(Pt=np.exp(-0.05/4), FT=95, K=80, t=1/4, T=1/2, sigmaS=0.2660, sigmaE=0.2490, sigmaF=0.0096, rhoSE=0.805, rhoSF=0.0805, rhoEF=0.1243, KappaE=1.045, KappaF=0.200).call() should equal 15.00468"

    assert (
        round(opt.put(), 6) == 0.191426
    ), "MiltersenSchwartzOption put price does not match fOptions. MiltersenSchwartzOption(Pt=np.exp(-0.05/4), FT=95, K=80, t=1/4, T=1/2, sigmaS=0.2660, sigmaE=0.2490, sigmaF=0.0096, rhoSE=0.805, rhoSF=0.0805, rhoEF=0.1243, KappaE=1.045, KappaF=0.200).put() should equal 0.191426"

    assert isinstance(
        opt.summary(printer=False), str
    ), "MiltersenSchwartzOption.summary() failed to produce string."

    assert isinstance(
        opt.get_params(), dict
    ), "MiltersenSchwartzOption.get_params failed to return dictionary of values"


def test_FDM_greeks():
    opt = fo.GBSOption(10.0, 8.0, 1, 0.02, 0.0, 0.1)
    greeks = ["delta", "theta", "gamma", "lamb", "vega", "rho"]

    for g in greeks:
        for call in [True, False]:
            greek_func = getattr(opt, g)
            if g not in ["gamma", "vega"]:
                test = np.allclose(
                    greek_func(call=call, method="analytic"),
                    greek_func(call=call, method="fdm"),
                    rtol=0.0001,
                )
            else:
                test = np.allclose(
                    greek_func(method="analytic"),
                    greek_func(method="fdm"),
                    rtol=0.0001,
                )

            assert (
                test
            ), f"FDM greek calc for {g} with call={call} did not match analytics solution from GBSOption. analytic={greek_func(method='analytic')}, fdm={greek_func(method='fdm')}"
