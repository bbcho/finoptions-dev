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


def test_fHN():
    test = fo.heston_nandi_options._fHN(
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
        real=True,
    )

    assert np.allclose(test, 0.01201465), "fstarHN failed test 1"

    test = fo.heston_nandi_options._fHN(
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
        real=True,
    )

    assert np.allclose(test, 0.0001524204), "fstarHN failed test 2"


def test_heston_option_value():

    opt = fo.heston_nandi_options.HestonNandiOption(
        S=S, K=K, t=t, r=r, lamb=lamb, omega=omega, alpha=alpha, beta=beta, gamma=gamma
    )

    test = opt.call()
    assert np.allclose(test, 8.9921), "Heston Option failed test 1"

    test = opt.put()
    assert np.allclose(test, 4.115042), "Heston Option failed test 2"

    opt = fo.heston_nandi_options.HestonNandiOption(
        S=S, K=90, t=t, r=r, lamb=lamb, omega=omega, alpha=alpha, beta=beta, gamma=gamma
    )

    test = opt.call()
    assert np.allclose(test, 15.85447), "Heston Option failed test 1"

    test = opt.put()
    assert np.allclose(test, 1.465121), "Heston Option failed test 2"


def test_heston_delta_gamma():

    opt = fo.heston_nandi_options.HestonNandiOption(
        S=S, K=K, t=t, r=r, lamb=lamb, omega=omega, alpha=alpha, beta=beta, gamma=gamma
    )

    test = opt.delta(call=True)
    assert np.allclose(test, 0.6739534), "Heston Greek failed test 1"

    test = opt.delta(call=False)
    assert np.allclose(test, -0.3260466), "Heston Greek failed test 2"

    test = opt.gamma()
    assert np.allclose(test, 0.02211149), "Heston Greek failed test 3"
