import sys
import os
import numpy as np


sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src/")

import energyderivatives as ed


def test_RollGeskeWhaleyOption():
    opt = ed.basic_american_options.RollGeskeWhaleyOption(
        S=80, K=82, t=1 / 3, td=1 / 4, r=0.06, D=4, sigma=0.30
    )

    assert (
        round(opt.call(), 5) == 4.38603
    ), "RollGeskeWhaleyOption call price does not match fOptions. RollGeskeWhaleyOption(S=80, K=82, t=1/3, td=1/4, r=0.06, D=4, sigma=0.30) should equal 4.38603"


def test_BAWAmericanApproxOption():
    opt = ed.basic_american_options.BAWAmericanApproxOption(
        S=100, K=90, t=0.5, r=0.10, b=0, sigma=0.25
    )

    assert (
        round(opt.call(), 5) == 12.44166
    ), "BAWAmericanApproxOption call price does not match fOptions. BAWAmericanApproxOption(S = 100, K = 90, t = 0.5, r = 0.10, b = 0, sigma = 0.25).call() should equal 12.44166"

    assert (
        round(opt.put(), 5) == 2.74361
    ), "BAWAmericanApproxOption put price does not match fOptions. BAWAmericanApproxOption(S = 100, K = 90, t = 0.5, r = 0.10, b = 0, sigma = 0.25).put() should equal 2.74361"


def test_BSAmericanApproxOption():

    opt = ed.basic_american_options.BSAmericanApproxOption(
        S=100, K=90, t=0.5, r=0.10, b=0, sigma=0.25
    )

    assert (
        round(opt.call()["OptionPrice"], 5) == 12.39889
    ), "BSAmericanApproxOption call price does not match fOptions. BSAmericanApproxOption(S=100, K=90, t=0.5, r=0.10, b=0, sigma=0.25).call()['OptionPrice'] should equal 12.39889"

    assert (
        round(opt.call()["TriggerPrice"], 5) == 115.27226
    ), "BSAmericanApproxOption call trigger price does not match fOptions. BSAmericanApproxOption(S=100, K=90, t=0.5, r=0.10, b=0, sigma=0.25).call()['TriggerPrice'] should equal 115.27226"

    assert (
        round(opt.put()["OptionPrice"], 6) == 2.714363
    ), "BSAmericanApproxOption put price does not match fOptions. BSAmericanApproxOption(S=100, K=90, t=0.5, r=0.10, b=0, sigma=0.25).call()['OptionPrice'] should equal 2.714363"

    assert (
        round(opt.put()["TriggerPrice"], 5) == 128.08029
    ), "BSAmericanApproxOption put trigger price does not match fOptions. BSAmericanApproxOption(S=100, K=90, t=0.5, r=0.10, b=0, sigma=0.25).call()['TriggerPrice'] should equal 128.08029"
