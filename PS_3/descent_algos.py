import numpy as np
import numpy.linalg as la
from utils import *

def descent(update, A, b, reg, T=int(1e4), c=1e-5):
    # Descends using update method "update" for T steps

    x = np.zeros(A.shape[1])
    xs = []
    error = [la.norm(np.dot(A, x) - b)]
    l1 = [np.sum(abs(x))]
    for t in range(T):
        # update A (either subgradient or frank-wolfe)
        x = update(x, A, b, t, reg, c)

        # record error and l1 norm

        if (t % 1 == 0) or (t == T - 1):
            error.append(la.norm(np.dot(A, x) - b))
            l1.append(np.sum(abs(x)))
            xs.append(x)
            assert not np.isnan(error[-1])

    return x, error, l1, xs

def proximal_gradient_update(x, A, b, t, lam, beta):
    # Updates x based on PGD
    gradient = get_l2_subgrad(x, A, b)
    eta = 1.0/beta
    x_bar = x - eta*gradient
    x_plus = soft_thresholding(x_bar, lam, eta)
    return x_plus


def FISTA_update(x, y, A, b, t, lam_t, beta_alpha):
    # Updates x based on PGD with Acceleration
    # (Implemented based on Bubeck's Notes)
    lam_plus = (1 + np.sqrt(1+4*lam_t**2))/2.0
    gamma_t = (1-lam_t)/lam_plus    #TODO: figure this out?
    gradient = get_l2_subgrad(x, A, b)
    beta, alpha = beta_alpha
    kappa = alpha/beta
    eta = 1.0/beta
    gamma = (np.sqrt(kappa)-1)/(np.sqrt(kappa)+1)

    y_bar = x - eta*gradient
    y_plus = soft_thresholding(y_bar, 1.0, eta)
    # x_plus = (1-gamma_t)*y_plus + gamma_t*y
    # x_plus = (1+gamma_t)*y_plus - gamma_t*y
    x_plus = (1+gamma)*y_plus - gamma*y

    print(np.linalg.norm(y_plus, order=1))
    return x_plus, y_plus, lam_plus


def frank_wolfe_update(x, A, b, t, gam, c):
    # Updates x using Frank-Wolfe method for loss ==> (1/2)*||Ax-b||_2^2
    # and constraint: {x | lam*||x||_1 <= gam}
    n_t = 2.0 / (t + 2.0)

    neg_g_t = -1.0 * get_l2_subgrad(x, A, b)

    s_t = np.zeros(np.shape(x))
    index = np.argmax(np.abs(neg_g_t))
    s_t[index] = gam * np.sign(neg_g_t[index])

    x = x + n_t * (s_t - x)

    return x


def frank_wolfe_update_btls(x, A, b, t, gam, c):
    # Updates x using Frank-Wolfe method for loss ==> (1/2)*||Ax-b||_2^2
    # and constraint: {x | lam*||x||_1 <= gam}
    # (Uses BTLS)
    n_t = 1.0
    tau = 1.0/2.0
    old_loss = get_l2_loss(x, A, b)
    new_loss = old_loss + 1.0
    x_plus = x

    # Calculate gradient and constrained gradient
    neg_g_t = -1.0 * get_l2_subgrad(x, A, b)
    s_t = np.zeros(np.shape(x))
    index = np.argmax(np.abs(neg_g_t))
    s_t[index] = gam * np.sign(neg_g_t[index])

    # BTLS Loop
    while new_loss > old_loss - 1/2*n_t*np.dot(neg_g_t.T, (s_t-x)):
        x_plus = x + n_t*(s_t-x)
        new_loss = get_l2_loss(x_plus, A, b)

        n_t = n_t*tau

    return x_plus


def subgradient_update(x, A, b, t, lam, c=1e-5):
    # Updates x using subgradient descent using loss ==> (1/2)*||Ax-b||_2^2 + lam*||x||_1
    n_t = c / np.sqrt(t + 1)

    subgrad = get_l2_subgrad(x, A, b) + lam * get_l1_subgrad(x)

    x = x - n_t * subgrad

    return x


def subgradient_update_btls(x, A, b, t, lam, c):
    # Updates x using subgradient descent using loss ==> (1/2)*||Ax-b||_2^2 + lam*||x||_1
    # (Uses BTLS)
    ticks = 0
    n_t = 1.0
    tau = 0.75
    old_loss = get_l2_loss(x, A, b) + lam*np.linalg.norm(x, ord=1)
    new_loss = old_loss + 1.0
    x_plus = x

    # Calculate gradient
    subgrad = get_l2_subgrad(x, A, b) + lam*get_l1_subgrad(x)

    # BTLS Loop
    while new_loss > old_loss - 1/2*n_t*np.square(la.norm(subgrad)):

        x_plus = x - n_t * subgrad
        new_loss = get_l2_loss(x_plus, A, b) + lam*np.linalg.norm(x_plus, ord=1)

        n_t = n_t*tau
        ticks = ticks + 1

    return x_plus



def accelerated_descent(update, A, b, reg, T=int(1e4), c=1e-5):
    # Descends using update method "update" for T steps

    x_t = np.zeros(A.shape[1])
    y_t = np.zeros(A.shape[1])
    lam_t = 0
    xs = []
    error = [la.norm(np.dot(A, x_t) - b)]
    l1 = [np.sum(abs(x_t))]
    for t in range(T):
        # update A (either subgradient or frank-wolfe)
        x_t1, y_t1, lam_t1 = update(x_t, y_t, A, b, t, lam_t, c)

        x_t = x_t1
        y_t = y_t1
        lam_t = lam_t1
        # record error and l1 norm

        if (t % 1 == 0) or (t == T - 1):
            error.append(la.norm(np.dot(A, x_t1) - b))
            l1.append(np.sum(abs(x_t1)))
            xs.append(x_t1)
            assert not np.isnan(error[-1])

    return x_t1, error, l1, xs