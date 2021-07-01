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
