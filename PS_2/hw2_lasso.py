import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


def frank_wolfe_update(x, A, b, t, gam, c):
    # Updates x using Frank-Wolfe method for loss ==> (1/2)*||Ax-b||_2^2
    # and constraint: {x | lam*||x||_1 <= gam}
    n_t = 2.0/(t+2.0)

    neg_g_t = -1.0*get_l2_subgrad(x, A, b)

    s_t = np.zeros(np.shape(x))
    index = np.argmax(np.abs(neg_g_t))
    s_t[index] = gam*np.sign(neg_g_t[index])

    x = x + n_t*(s_t-x)

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
    n_t = c/np.sqrt(t+1)

    subgrad = get_l2_subgrad(x, A, b) + lam*get_l1_subgrad(x)

    x = x - n_t*subgrad

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

def get_l2_loss(x, A, b):

    return 1.0/2.0*np.linalg.norm(np.matmul(A, x) - b)**2

def get_l1_subgrad(x):
    # Returns subgradient for l1 loss ==> ||x||_1

    return np.sign(x)

def get_l2_subgrad(x, A, b):
    # Returns subgradient for l2 loss ==> (1/2)*||Ax-b||_2^2
    return np.matmul(np.transpose(A), np.matmul(A, x) - b)

def descent(update, A, b, reg, T=int(1e4), c=1e-5):
    # Descends using update method "update" for T steps

    x = np.zeros(A.shape[1])
    error = [la.norm(np.dot(A, x) - b)]
    l1 = [np.sum(abs(x))]
    for t in range(T):
        # update A (either subgradient or frank-wolfe)
        x = update(x, A, b, t, reg, c)
        
        # record error and l1 norm
        if (t % 1 == 0) or (t == T - 1):
            error.append(la.norm(np.dot(A, x) - b))
            l1.append(np.sum(abs(x)))

            assert not np.isnan(error[-1])

    return x, error, l1


def main(T=int(1e4)):

    A = np.load("A.npy")
    b = np.load("b.npy")

    # LASSO using subgradient and FW methods
    x_sg, error_sg, l1_sg = descent(subgradient_update, A, b, reg=5, T=T, c=1e-5)
    norm_star = l1_sg[-1]
    x_fw, error_fw, l1_fw = descent(frank_wolfe_update, A, b, reg=norm_star, T=T)

    # LASSO using subgradient and FW methods + BTLS
    x_sg_btls, error_sg_btls, l1_sg_btls = descent(subgradient_update_btls, A, b, reg=1, T=T, c=1e-5)
    norm_star = l1_sg_btls[-1]
    x_fw_btls, error_fw_btls, l1_fw_btls = descent(frank_wolfe_update_btls, A, b, reg=norm_star, T=T)

    # Plots
    plt.clf()
    plt.plot(error_sg, label='Subgradient')
    plt.plot(error_fw, label='Frank-Wolfe')
    plt.title('Error')
    plt.legend()
    plt.savefig('error.eps')

    plt.clf()
    plt.plot(l1_sg, label='Subgradient')
    plt.plot(l1_fw, label='Frank-Wolfe')
    plt.title("$\ell^1$ Norm")
    plt.legend()
    plt.savefig('l1.eps')

    # Plots + BTLS
    plt.clf()
    plt.plot(error_sg_btls, label='Subgradient + BTLS')
    plt.plot(error_fw_btls, label='Frank-Wolfe + BTLS')
    plt.title('Error')
    plt.legend()
    plt.savefig('error_btls.eps')

    plt.clf()
    plt.plot(l1_sg_btls, label='Subgradient + BTLS')
    plt.plot(l1_fw_btls, label='Frank-Wolfe + BTLS')
    plt.title("$\ell^1$ Norm")
    plt.legend()
    plt.savefig('l1_btls.eps')


if __name__ == "__main__":
    main()