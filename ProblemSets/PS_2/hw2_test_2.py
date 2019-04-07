# ==========================
# Hyperparameter Tuning
# ==========================


import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from ProblemSets.utils import fuzzy_equal

# TODO: REMOVE THIS----------------------
from sklearn.linear_model import Lasso


# TODO: REMOVE THIS----------------------

def frank_wolfe(x, A, b, t, gam, c):
    # update x (your code here)
    return x


def subgradient(x, A, b, t, lam, c=1e-5):
    n_t = c / np.sqrt(t + 1)
    subgrad_l2 = np.matmul(np.transpose(A), np.matmul(A, x) - b)
    subgrad_l1 = get_l1_subgrad(x)

    subgrad = subgrad_l2 + lam * subgrad_l1

    x = x - n_t * subgrad

    return x


# add BTLS variants and include them in main/descent below

def get_l1_subgrad(x):
    l1_subgrad = np.empty([0, ])

    for element in x:
        if element == 0.0:
            # l1_subgrad.append(np.random.uniform(-1.0, 1.0))
            l1_subgrad = np.append(l1_subgrad, 0.0)

        else:
            l1_subgrad = np.append(l1_subgrad, np.sign(element))

    return l1_subgrad


def descent(update, A, b, reg, T=int(1e4), c=1e-5):
    x = np.zeros(A.shape[1])
    error = []
    l1 = []
    for t in range(T):
        # update A (either subgradient or frank-wolfe)
        x = update(x, A, b, t, reg, c)

        # record error and l1 norm
        if (t % 1 == 0) or (t == T - 1):
            error.append(la.norm(np.dot(A, x) - b))
            l1.append(np.sum(abs(x)))

            assert not np.isnan(error[-1])

    return x, error, l1


def main(T=int(1e3)):
    A = np.load("A.npy")
    b = np.load("b.npy")

    # hyperparameter tuning (stepsize and lambda)

    errors = {}

    for c in np.logspace(-8, -3, 10):
        for reg in np.logspace(-5, 2, 10):
            print(c)
            print(reg)
            # modify regularization parameters below
            x_sg, error_sg, l1_sg = descent(subgradient, A, b, reg=1, T=T, c=c)
            # x_fw, error_fw, l1_fw = descent(frank_wolfe, A, b, reg=0., T=T)
            # add BTLS experiments

            errors[(c, reg)] = error_sg[-1]

    best_pt = min(errors, key=errors.get)
    print('best pt:', best_pt)

    # add plots for BTLS
    plt.clf()
    plt.plot(error_sg, label='Subgradient')
    # plt.plot(error_fw, label='Frank-Wolfe')
    plt.title('Error')
    plt.legend()
    plt.savefig('error.eps')

    plt.clf()
    plt.plot(l1_sg, label='Subgradient')
    # plt.plot(l1_fw, label='Frank-Wolfe')
    plt.title("$\ell^1$ Norm")
    plt.legend()
    plt.savefig('l1.eps')

    # TODO: REMOVE THIS----------------------
    print('l2_error:', error_sg[-1])
    print('l0_norm:', np.sum([1 for elem in x_sg if not fuzzy_equal(abs(elem), 0.0, tol=1e-3)]))
    print('l1_norm:', np.sum(abs(x_sg)))
    print('------------------------------------------')

    lasso = Lasso(alpha=5e-2, random_state=0)
    lasso.fit(A, b)
    b_hat = lasso.predict(A)
    error = la.norm(b_hat - b)
    print('l2_error:', error)
    print('l0_norm:', np.sum([1 for elem in lasso.coef_ if elem != 0.0]))
    print('l1_norm:', np.sum(abs(lasso.coef_)))
    print(x_sg)
    plt.figure()
    plt.plot(x_sg)
    plt.show()


# TODO: REMOVE THIS----------------------

if __name__ == "__main__":
    main()