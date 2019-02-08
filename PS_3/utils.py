import numpy as np


def soft_thresholding(x, lam, step):
    # Conducts soft thresholding for ISTA
    zeros = np.zeros(np.shape(x))
    x_ = np.maximum((np.abs(x)-lam*step), zeros)*np.sign(x)
    return x_


def get_l1_subgrad(x):
    # Returns subgradient for l1 loss ==> ||x||_1
    l1_subgrad = np.empty([0, ])

    for element in x:
        if element == 0.0:
            # l1_subgrad.append(np.random.uniform(-1.0, 1.0))
            l1_subgrad = np.append(l1_subgrad, 0.0)

        else:
            l1_subgrad = np.append(l1_subgrad, np.sign(element))

    return l1_subgrad


def get_l2_subgrad(x, A, b):
    # Returns subgradient for l2 loss ==> (1/2)*||Ax-b||_2^2

    return np.matmul(np.transpose(A), np.matmul(A, x) - b)


def get_l2_loss(x, A, b):
    norm = np.linalg.norm(np.matmul(A, x) - b)
    return norm


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
