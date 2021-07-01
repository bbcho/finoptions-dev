import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src/")

import energyderivatives as ed


def test_Option():
    b = ed.Option(0, 0, 0, 0, 0, 0)

    assert isinstance(b, ed.Option), "Option object failed init"

    assert isinstance(
        b.get_params(), dict
    ), "Option.get_params failed to return dictionary of values"


def test_GBSOption():

    opt = ed.GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1)

    assert (
        round(opt.call(), 6) == 2.061847
    ), "GBSOption call price does not match fOptions. GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1).call() should equal 2.061847"

    assert (
        round(opt.put(), 8) == 0.00293811
    ), "GBSOption put price does not match fOptions. GBSOption(10.0, 8.0, 1.0, 0.02, 0.01, 0.1).put() should equal 0.00293811"

