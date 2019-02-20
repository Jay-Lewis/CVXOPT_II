import numpy as np
import torch
from torch.autograd import Variable
import random

# --------------------------
# Subgradients / Gradients
# --------------------------



# --------------------------
# Losses
# --------------------------

def l2_loss_scaled(x, data, params):

    A = data['A']
    b = data['b'].view(-1,1)
    b_hat = torch.mm(A, x).view(-1, 1)
    loss = torch.sqrt(torch.sum((b_hat-b)**2))
    # loss_fn = torch.nn.MSELoss()
    # loss = loss_fn(b_hat, b)

    return loss

def l1_loss_scaled(x, data, params):
    A = data['A']
    b = data['b'].view(-1,1)
    b_hat = torch.mm(A, x).view(-1, 1)
    loss = torch.sum(torch.abs(b_hat-b))

    return loss

def l1_loss(x, data, params):
    loss = torch.sum(x)

    return loss

def logist_loss(Beta, data, params):
    A = data['A']
    b = data['b'].view(-1)
    loss_fn = torch.nn.NLLLoss()
    m = torch.nn.LogSoftmax()
    loss = loss_fn(m(torch.mm(A, Beta)), b)
    if 'lam' in params:
        loss += Beta.norm(2)*params['lam']

    return loss

def logist_loss_(Beta, data, params):
    A, b = get_args_from_dict(data, ('A', 'b'))

    loss = 0

    for i, y_i in enumerate(b):
        x_i = A[i, :]
        B_i = Beta[:, int(y_i)]
        loss += torch.dot(B_i, x_i)

        summ = torch.sum(torch.exp(-1.0*torch.mm(x_i.view(1,-1), Beta)))
        loss += torch.log(summ)

    return loss

def frobenius_norm(X, data, params):

    return torch.norm(X, p=2)

def nuclear_norm(X, data, params):

    return torch.norm(X, p='nuc')

def matrix_rank(X, data, parms):

    return np.linalg.matrix_rank(to_numpy(X))

# --------------------------
# Projections
# --------------------------

def proj_prob_simp(y, data, params):
    #TODO: implement in Torch
    y = np.reshape(y, [-1])
    a = np.ones(np.shape(y))
    l = y / a
    idx = np.argsort(l)
    d = len(l)

    evalpL = lambda k: np.sum(a[idx[k:]] * (y[idx[k:]] - l[idx[k]] * a[idx[k:]])) - 1

    def bisectsearch():
        idxL, idxH = 0, d - 1
        L = evalpL(idxL)
        H = evalpL(idxH)

        if L < 0:
            return idxL

        while (idxH - idxL) > 1:
            iMid = int((idxL + idxH) / 2)
            M = evalpL(iMid)

            if M > 0:
                idxL, L = iMid, M
            else:
                idxH, H = iMid, M

        return idxH

    k = bisectsearch()
    lam = (np.sum(a[idx[k:]] * y[idx[k:]]) - 1) / np.sum(a[idx[k:]])

    x = np.maximum(0, y - lam * a)

    return np.reshape(x, [-1,1])

def proj_matrix_sample(X, data, params):
    O, M_samp = get_args_from_dict(data, ('O', 'M_samp'))

    return X - X*O + M_samp

# --------------------------
# Step Size functions
# --------------------------

def one_by_t(x, t, data, params):
    return 1.0/ t

def one_by_sqrt(x, t, data, params):
    return 1.0/ np.sqrt(t)


# --------------------------
# Helper functions
# --------------------------


def test_error(xs, loss_fn, data, params):
    test_errors = []
    for x in xs:
        loss = loss_fn(x, data, params)
        test_errors.append(loss)

    return test_errors


def sgd_matrix(A, indices, data, params):
    if indices is None:
        sgd_num = params['sgd_num']
        m = np.shape(A)[0]
        indices = random.sample(set(range(0, m)), sgd_num)
    A = A[indices]

    return A, indices

def sgd_vector(subgrad, indices, data, params):
    if indices is None:
        sgd_num = params['sgd_num']
        m = subgrad.shape[0]
        indices = random.sample(set(range(0, m)), sgd_num)
    stoch_subgrad = subgrad[indices, :]

    return stoch_subgrad

def get_alpha_beta(A):
    A = A.data
    u, s, vh = np.linalg.svd(A)
    alpha = min(np.abs(s))**2
    beta = max(np.abs(s))**2
    return alpha, beta


def to_sparse_tensor(coo_mat):
    values = coo_mat.data
    indices = np.vstack((coo_mat.row, coo_mat.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo_mat.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

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

def to_numpy(tensor):
    data = tensor.data
    if data.is_cuda:
        numpy = data.cpu().numpy()
    else:
        numpy = data.numpy()
    return numpy


def df_str_to_matrix(df):
    ''' Takes df with columns which have space separated string data
        and returns matrix of numbers '''

    matrix = []
    for row in df.values:
        new_row = row[0].split(' ')
        new_row = [float(num) for num in new_row]
        matrix.append(new_row)

    return np.asarray(matrix)