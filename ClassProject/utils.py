import numpy as np
from scipy.sparse.linalg import arpack

def basis_sampler(pi_s):
    # gives stand. basis vect e_i with prob p_i
    d = np.size(pi_s)
    uni = np.random.uniform()
    no_sample = True
    i = 0

    while no_sample:
        uni = uni - pi_s[i]
        if (uni < 0):
            choice = i
            no_sample = False

        i = i + 1

    sample = np.zeros(d)
    sample[choice] = 1

    return sample


def isPSD(A, tol = 1e-8):
    vals, vecs = arpack.eigsh(A, k = 2, which='BE') # return the ends of spectrum of A
    return np.all(vals > -tol)


def kappa_line_search(C_hat, H, kappa_start=1, tol=1e-3):
    kappa = kappa_start
    kappa_prev = kappa
    best_NO = 0
    best_YES = None
    i = 0

    while True:
        if isPSD(kappa*H-C_hat):
            best_YES = kappa
        else:
            best_NO = kappa

        if best_YES is None:
            kappa_prev = kappa
            kappa = 2*best_NO
        else:
            kappa_prev = kappa
            kappa = (best_YES + best_NO) / 2


        if np.abs(kappa-kappa_prev) < tol and best_YES is not None:
            return best_YES

        if i % 10 == 0: print('Iteration: ', i)
        i = i + 1