import sys
import os
import numpy as np


sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src/")

import finoptions as fo

phi = 20
# const = 1
lamb = -0.5
omega = 2.3e-6
alpha = 2.9e-6
beta = 0.85
gamma = 184.25
S = 100
K = 100
t = 252
r = 0.05 / t


def test_fstarHN():
    test = fo.heston_nandi_options._fstarHN(
        phi=phi,
        const=1,
        lamb=lamb,
        omega=omega,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        S=S,
        K=K,
        t=t,
        r=r,
    )

    assert np.allclose(test, 0.01201465), "fstarHN failed test 1"

    test = fo.heston_nandi_options._fstarHN(
        phi=phi,
        const=0,
        lamb=lamb,
        omega=omega,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        S=S,
        K=K,
        t=t,
        r=r,
    )

    assert np.allclose(test, 0.0001524204), "fstarHN failed test 2"
