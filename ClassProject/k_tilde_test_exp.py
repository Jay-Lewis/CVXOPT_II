# ===========
# Imports
# ===========

import numpy as np
import matplotlib.pyplot as plt
import utils


# =======================
# Test 1: Uncorrelated Gaussian
# =======================
kappas = []
kappa_tildes = []

# ------- LOOP dimension-------
dims = [int(d) for d in np.linspace(2, 20, 6)]
for d in dims:

    # ------- Create Estimate of Norm Weighted Cov. Matrix -------
    n = 10000
    secnd_moments = np.abs(np.random.randn(d))
    cov = np.diag(secnd_moments)
    mu = np.zeros(d)
    estimate = np.zeros([d, d])

    for _ in range(0, n):
        a = np.random.multivariate_normal(mu, cov)
        norm = np.inner(a, a)
        matrix = np.outer(a, a)
        estimate = estimate + norm*matrix

    # ------- Find Kappa -------
    # Do BTLS on Kappa to find approximation
    # start with sigma_max^2/sigma_min^2

    kappa_start = np.max(np.diagonal(cov))/np.min(np.diagonal(cov))
    R_squared = utils.kappa_line_search(estimate, cov, kappa_start)
    u = np.min(np.diagonal(cov))
    kappa = R_squared/u
    kappas.append(kappa)
    print('kappa:', kappa)



    # ------- Create Estimate of Norm Weighted (H^-1) Cov. Matrix -------

    secnd_moments = np.random.randn(d)**2
    cov = np.diag(secnd_moments)
    mu = np.zeros(d)
    estimate = np.zeros([d, d])

    for _ in range(0, n):
        a = np.random.multivariate_normal(mu, cov)
        norm = norm = np.matmul(a, np.matmul(np.linalg.inv(cov), a))
        matrix = np.outer(a, a)
        estimate = estimate + norm*matrix

    # ------- Find Kappa Tilde -------
    # Do BTLS on Kappa to find approximation
    # start with d

    kappa_start = d
    kappa_tilde = utils.kappa_line_search(estimate, cov, d)
    kappa_tildes.append(kappa_tilde)
    print('kappa_tilde:', kappa_tilde)


# ------- Plot Kappas vs. Dimension -------

plt.plot(dims, kappas, label='Kappa')
plt.plot(dims, kappa_tildes, label='Kappa_tilde')
plt.show()
plt.legend()

# =======================
# Test 2: Correlated Gaussian
# =======================


# =======================
# Test 3: Rayleigh + Laplace
# =======================