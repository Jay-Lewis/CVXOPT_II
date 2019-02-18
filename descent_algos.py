import numpy as np
import numpy.linalg as la
import utils
import torch

class descent_structure:
    def __init__(self, data, parameters, loss_fn):
        self.data = data
        self.parameters = parameters
        self.loss_fn = loss_fn
        if 'save_int' in parameters:
            self.save_int = parameters['save_int']
        else:
            self.save_int = 1

    def descent(self, update_fn, subgrad_fn, norm_fn, x_start):
        # Descends using update method "update" for T steps

        T = self.parameters['T']
        x = x_start
        xs = []
        error = []
        if norm_fn is not None:
            norm = [utils.to_numpy(norm_fn(x, self.data, self.parameters))]
        else:
            norm = None

        for t in range(T):
            # update x
            x.data = update_fn(x, t, subgrad_fn, self.data, self.parameters)

            # record error and norm (if required)

            if (t % self.save_int == 0) or (t == T - 1):
                error.append(utils.to_numpy(self.loss_fn(x, self.data, self.parameters)))
                if norm_fn is not None:
                    norm.append(utils.to_numpy(norm_fn(x, self.data, self.parameters)))
                xs.append(x.data)
                assert not np.isnan(error[-1])

        return x, error, norm, xs

    def accelerated_descent(self, update_fn, subgrad_fn, error_fn, norm_fn, x_start):
        # TODO: switch to torch
        # Descends using update method "update" for T steps
        T = self.parameters['T']
        x_t = x_start
        y_t = x_start

        lam_t = 0
        xs = []
        error = [error_fn(x_start, self.data, self.parameters)]
        if norm_fn is not None:
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

    def get_torch_subgrad(self, x, data, params):
        if x.grad is not None:
            x.grad.zero_()
        loss = self.loss_fn(x, data, params)
        loss.backward()
        subgrad = x.grad

        return subgrad

# ========================
# Standard Methods
# ========================

def subgradient_update(x, t, subgradient_fn, data, params):
    # Updates x using subgradient descent, subgradients from subgradient_fn

    if 'c' in params:
        c = params['c']
        n_t = c / np.sqrt(t + 1)
    else:
        n_t = 1.0 / params['beta']

    subgrad = subgradient_fn(x, data, params)
    x = x - subgrad*n_t

    return x

def project_gradient_update(x, t, subgradient_fn, data, params):
    beta, dtype = utils.get_args_from_dict(params, ('beta', 'dtype'))
    eta = 1.0 / beta
    project_fn = params['project_fn']
    gradient = subgradient_fn(x, data, params)
    x = x - gradient*eta
    x.data = torch.Tensor(project_fn(utils.to_numpy(x), data, params)).type(dtype)

    return x

def mirror_descent_KL_prob(x, t, subgradient_fn, data, params):
    # With Bregman Divergence = KL_div and set is probability simplex,
    #  update is normalized exponential gradient
    n_t = 1.0 / params['beta']

    subgrad = subgradient_fn(x, data, params)
    x = x*torch.exp(subgrad*-n_t)/torch.sum(x*torch.exp(subgrad*-n_t))
    return x

# ========================
# Stochastic  Methods
# ========================

def data_stoch_sg_update(x, t, subgradient_fn, data, params):
    # Updates x using subgradient descent, subgradients from subgradient_fn
    # stochastic in the number of samples used for gradient

    if 'c' in params:
        c = params['c']
        n_t = c / np.sqrt(t + 1)
    else:
        n_t = 1.0 / params['beta']

    data['A'], indices = utils.sgd_matrix(data['A_save'], None, data, params)
    data['b'] = utils.sgd_vector(data['b_save'], indices, data, params)
    subgrad_sgd = subgradient_fn(x, data, params)
    x = x - subgrad_sgd*n_t
    data['A'] = data['A_save']
    data['b'] = data['b_save']

    return x

def svrg_update(x, t, subgradient_fn, data, params):
    # Updates x using svrg descent, subgradients from subgradient_fn
    # stochastic in the number of samples used for gradient
    if 'c' in params:
        c = params['c']
        n_t = c / np.sqrt(t + 1)
    else:
        n_t = 1.0 / params['beta']

    # Occasionally compute new full subgradient
    svrg_s = params['svrg_s']
    if t % svrg_s == 0:
        subgrad = subgradient_fn(x, data, params)
        data['svrg_save'] = x
        data['svrg_grad_save'] = subgrad

    # Compute stochastic subgradient
    data['A'], indices = utils.sgd_matrix(data['A_save'], None, data, params)
    data['b'] = utils.sgd_vector(data['b_save'], indices, data, params)
    y = data['svrg_save']; y_subgrad = data['svrg_grad_save']

    subgrad_sgd = subgradient_fn(x, data, params)
    subgrad_sgd_2 = subgradient_fn(y, data, params)
    subgrad_full = subgrad_sgd - subgrad_sgd_2 + y_subgrad

    x = x - subgrad_full*n_t
    data['A'] = data['A_save']
    data['b'] = data['b_save']

    return x


# ========================
# Accelerated Methods
# =======================

# def accelerated_subgrad_update(x_t, y, lam_t, subgradient_fn, data, params):
#     # Updates x based on SubGrad with Nesterov Acceleration
#     # TODO: switch to torch
#
#     alpha, beta = get_args_from_dict(params, ('alpha', 'beta'))
#
#     eta = 1.0/beta
#     kappa = alpha/beta
#     gamma = (1-np.sqrt(kappa))/(np.sqrt(kappa)+1)
#
#     gradient = subgradient_fn(y, data, params)
#     x_plus = y - eta*gradient
#     y_plus = x_plus + gamma*(x_plus-x_t)
#
#     return x_plus, y_plus, None
