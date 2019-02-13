import numpy as np


# --------------------------
# Subgradients / Gradients
# --------------------------

def get_l1_subgrad(x, data=None, params=None):
    # Returns subgradient for l1 loss ==> ||x||_1
    return np.sign(x)


def get_l2_subgrad(x, data, params):
    # Returns subgradient for l2 loss ==> (1/2)*||Ax-b||_2^2
    A, b = get_args_from_dict(data, ('A', 'b'))

    return np.matmul(np.transpose(A), np.matmul(A, x) - b)


def get_LASSO_subgrad(x, data, params):
    lam = params['lam']
    subgrad = get_l2_subgrad(x, data, params) + lam * get_l1_subgrad(x)
    return subgrad


def get_logist_subgrad(Beta, data, params):
    X, y = get_args_from_dict(data, ('X', 'y'))
    mu = params['mu']
    num_c = Beta.shape[1]
    subgrad = np.zeros(np.shape(Beta))

    for i, y_i in enumerate(y):
        x_i = X[:, i]

        exp_vect = np.exp(-1.0*np.matmul(Beta.T, x_i))
        norm_factor = np.sum(exp_vect)

        for k in range(0, num_c):
            if k == y_i:
                subgrad[:, k] += (1.0-exp_vect[k]/norm_factor)*x_i
            else:
                subgrad[:, k] += (-exp_vect[k]/norm_factor)*x_i

    subgrad = subgrad + 2*mu*Beta

    return subgrad

# --------------------------
# Losses
# --------------------------

def get_l2_loss(x, data, params):
    A, b = get_args_from_dict(data, ('A', 'b'))
    norm = np.linalg.norm(np.matmul(A, x) - b)
    return norm


def get_l1_loss(x, data, params):
    norm = np.linalg.norm(x, ord=1)
    return norm


def get_logist_loss(Beta, data, params):
    X, y = get_args_from_dict(data, ('X', 'y'))

    loss = 0

    for i, y_i in enumerate(y):
        x_i = X[:, i]
        B_i = Beta[:, int(y_i)]
        loss += np.dot(B_i, x_i)

        summ = np.sum(np.exp(-1.0*np.matmul(Beta.T, x_i)))
        loss += np.log(summ)

    return loss


# --------------------------
# Other
# --------------------------

def soft_thresholding(x, lam, step):
    # Conducts soft thresholding for ISTA
    zeros = np.zeros(np.shape(x))
    x_ = np.maximum((np.abs(x)-lam*step), zeros)*np.sign(x)
    return x_

def get_logistic_preds(X, Beta):
    logits = []

    for x in X.T:
        logit = np.argmin(np.matmul(Beta.T, x))
        logits.append(logit)

    return logits


# --------------------------
# Helper functions
# --------------------------


def test_error(xs, data, params):
    test_errors = []
    for x in xs:
        loss = get_l2_loss(x, data, params)
        test_errors.append(loss)

    return test_errors


def get_alpha_beta(A):
    u, s, vh = np.linalg.svd(A)
    alpha = min(np.abs(s))**2
    beta = max(np.abs(s))**2
    return alpha, beta


def get_args_from_dict(args_dict, kw_list):
    args = ()

    for kw in kw_list:
        args += (args_dict[kw],)

    return args


def get_args_dict(kws, keys):
    dict = {}
    for kw, key in zip(kws, keys):
        dict[kw] = key

    return dict


# =========================================
# Non-Core code
# =========================================

def get_prob3_subgrad(x, data, params):
    # Returns subgradient for log. loss ==> log(sum_i(exp(ai^T*x-bi)))
    A, b = get_args_from_dict(data, ('A', 'b'))

    A_x = np.matmul(A, x)
    A_x_pb = A_x + b

    expon = np.exp(A_x_pb)
    sum1 = np.sum(expon)

    subgrad = np.matmul(A.T, expon)/sum1

    return subgrad

def get_prob3_loss(x, data, params):
    A, b = get_args_from_dict(data, ('A', 'b'))
    A_x = np.matmul(A, x)
    A_x_pb = A_x + b
    expon = np.exp(A_x_pb)
    loss = np.log(np.sum(expon))
    return loss