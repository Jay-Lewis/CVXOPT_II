import numpy as np
import numpy.linalg as la
from utils import *

class descent_structure:
    def __init__(self, data, parameters):
        self.data = data
        self.parameters = parameters

    def descent(self, update_fn, subgrad_fn, error_fn, norm_fn, x_start):
        # Descends using update method "update" for T steps
        T = self.parameters['T']
        x = x_start
        xs = []
        error = [error_fn(x, self.data, self.parameters)]
        norm = [norm_fn(x, self.data, self.parameters)]

        for t in range(T):
            # update A (either subgradient or frank-wolfe)
            x = update_fn(x, t, subgrad_fn, self.data, self.parameters)

            # record error and l1 norm

            if (t % 1 == 0) or (t == T - 1):
                error.append(error_fn(x, self.data, self.parameters))
                if norm_fn is not None:
                    norm.append(norm_fn(x, self.data, self.parameters))
                xs.append(x)
                assert not np.isnan(error[-1])

        return x, error, norm, xs

    def accelerated_descent(self, update_fn, subgrad_fn, error_fn, norm_fn, x_start):
        # Descends using update method "update" for T steps
        T = self.parameters['T']
        x_t = x_start
        y_t = x_start

        lam_t = 0
        xs = []
        error = [error_fn(x_start, self.data, self.parameters)]
        norm = [norm_fn(x_start, self.data, self.parameters)]

        for t in range(T):
            # update A (either subgradient or frank-wolfe)
            x_plus, y_plus, lam_plus = update_fn(x_t, y_t, lam_t, subgrad_fn, self.data, self.parameters)

            x_minus = x_t
            x_t = x_plus
            y_t = y_plus
            lam_t = lam_plus

            # record error and l1 norm
            if (t % 1 == 0) or (t == T - 1):
                error.append(error_fn(x_plus, self.data, self.parameters))
                norm.append(norm_fn(x_plus, self.data, self.parameters))
                xs.append(x_plus)
                assert not np.isnan(error[-1])

        return x_plus, error, norm, xs

# def proximal_gradient_update(subgradient_fn, x, A, b, t, lam, beta):
def proximal_gradient_update(x, t, subgradient_fn, data, params):
    # Updates x based on PGD
    beta, lam = get_args_from_dict(params, ('beta', 'lam'))
    gradient = subgradient_fn(x, data, params)
    eta = 1.0/beta
    x_bar = x - eta*gradient
    x_plus = soft_thresholding(x_bar, lam, eta)
    return x_plus


def FISTA_update(x_t, y, lam_t, subgradient_fn, data, params):
    # Updates x based on PGD with Acceleration
    # (Implemented based on Bubeck's Notes)
    alpha, beta = get_args_from_dict(params, ('alpha', 'beta'))

    lam_plus = (1 + np.sqrt(1+4*lam_t**2))/2.0
    gamma_t = (lam_t-1)/lam_plus
    eta = 1.0/beta

    gradient = subgradient_fn(y, data, params)
    x_bar = y - eta*gradient
    x_plus = soft_thresholding(x_bar, 1.0, eta)
    y_plus = x_plus + gamma_t*(x_plus-x_t)

    return x_plus, y_plus, lam_plus

def frank_wolfe_update(x, t, subgradient_fn, data, params):
    # Updates x using Frank-Wolfe method for loss ==> (1/2)*||Ax-b||_2^2
    # and constraint: {x | lam*||x||_1 <= gam}

    gamma = params['gamma']
    n_t = 2.0 / (t + 2.0)

    neg_g_t = -1.0*subgradient_fn(x, data, params)

    s_t = np.zeros(np.shape(x))
    index = np.argmax(np.abs(neg_g_t))
    s_t[index] = gamma * np.sign(neg_g_t[index])

    x = x + n_t * (s_t - x)

    return x


def subgradient_update(x, t, subgradient_fn, data, params):
    # Updates x using subgradient descent using loss ==> (1/2)*||Ax-b||_2^2 + lam*||x||_1

    if 'c' in params:
        c = params['c']
        n_t = c / np.sqrt(t + 1)
    else:
        n_t = 1.0 / params['beta']

    subgrad = subgradient_fn(x, data, params)

    x = x - n_t * subgrad

    return x

def accelerated_subgrad_update(x_t, y, lam_t, subgradient_fn, data, params):
    # Updates x based on SubGrad with Nesterov Acceleration
    alpha, beta = get_args_from_dict(params, ('alpha', 'beta'))

    eta = 1.0/beta
    kappa = alpha/beta
    gamma = (1-np.sqrt(kappa))/(np.sqrt(kappa)+1)

    gradient = subgradient_fn(y, data, params)
    x_plus = y - eta*gradient
    y_plus = x_plus + gamma*(x_plus-x_t)

    return x_plus, y_plus, None

def subgradient_update_btls(x, t, subgradient_fn, data, params):
    # Updates x using subgradient descent using loss ==> (1/2)*||Ax-b||_2^2 + lam*||x||_1
    # (Uses BTLS)
    A, b = get_args_from_dict(data, ('A', 'b'))
    lam = params['lam']
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

def frank_wolfe_update_btls(x, t, subgradient_fn, data, params):
    # Updates x using Frank-Wolfe method for loss ==> (1/2)*||Ax-b||_2^2
    # and constraint: {x | lam*||x||_1 <= gam}
    # (Uses BTLS)

    gamma = params['gamma']

    n_t = 1.0
    tau = 1.0/2.0
    old_loss = get_l2_loss(x, data, params)
    new_loss = old_loss + 1.0
    x_plus = x

    # Calculate gradient and constrained gradient
    neg_g_t = -1.0 * subgradient_fn(x, data, params)
    s_t = np.zeros(np.shape(x))
    index = np.argmax(np.abs(neg_g_t))
    s_t[index] = gamma * np.sign(neg_g_t[index])

    # BTLS Loop
    while new_loss > old_loss - 1/2*n_t*np.dot(neg_g_t.T, (s_t-x)):
        x_plus = x + n_t*(s_t-x)
        new_loss = get_l2_loss(x_plus, data, params)

        n_t = n_t*tau

    return x_plus