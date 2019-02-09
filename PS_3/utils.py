import numpy as np


def soft_thresholding(x, lam, step):
    # Conducts soft thresholding for ISTA
    zeros = np.zeros(np.shape(x))
    x_ = np.maximum((np.abs(x)-lam*step), zeros)*np.sign(x)
    return x_


def get_l1_subgrad(x, data=None, params=None):
    # Returns subgradient for l1 loss ==> ||x||_1
    return np.sign(x)


def get_l2_subgrad(x, data, params):
    # Returns subgradient for l2 loss ==> (1/2)*||Ax-b||_2^2
    A, b = get_args_from_dict(data, ('A', 'b'))

    return np.matmul(np.transpose(A), np.matmul(A, x) - b)


def get_logist_subgrad(x, data, params):
    # Returns subgradient for log. loss ==> log(sum_i(exp(ai^T*x-bi)))
    A, b = get_args_from_dict(data, ('A', 'b'))

    A_x = np.matmul(A, x)
    sum1 = 0; sum2 = 0
    n = np.size(x)
    subgrad = np.zeros([1, n])

    for j in range(0, n):
        for i, (inner_prod, b_i) in enumerate(zip(A_x, b)):
            a_j = A[i, j]
            sum1 += np.exp(inner_prod+b_i)
            sum2 += np.exp(inner_prod+b_i)*a_j

        subgrad[0, j] = sum2/sum1

    return subgrad


def get_LASSO_subgrad(x, data, params):
    lam = params['lam']
    subgrad = get_l2_subgrad(x, data, params) + lam * get_l1_subgrad(x)
    return subgrad


def get_l2_loss(x, A, b):
    norm = np.linalg.norm(np.matmul(A, x) - b)
    return norm


def get_logist_loss(x, A, b):
    sum = 0
    A_x = np.matmul(A, x)

    for inner_prod, b_i in zip(A_x, b):
        sum += np.exp(inner_prod+b_i)

    loss = np.log(sum)
    return loss


def test_error(xs, A, b):
    test_errors = []
    for x in xs:
        loss = get_l2_loss(x, A, b)
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
