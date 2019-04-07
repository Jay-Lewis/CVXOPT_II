import matplotlib.pyplot as plt
from ProblemSets.descent_algos import *
from ProblemSets import utils

# def frank_wolfe_update(x, A, b, t, gam, c):
#     # Updates x using Frank-Wolfe method for loss ==> (1/2)*||Ax-b||_2^2
#     # and constraint: {x | lam*||x||_1 <= gam}
#     n_t = 2.0/(t+2.0)
#
#     neg_g_t = -1.0*get_l2_subgrad(x, A, b)
#
#     s_t = np.zeros(np.shape(x))
#     index = np.argmax(np.abs(neg_g_t))
#     s_t[index] = gam*np.sign(neg_g_t[index])
#
#     x = x + n_t*(s_t-x)
#
#     return x


# ==========================================
# HW 3 (Problem 1)
# ==========================================

# --------------------
# Fix Random Seed
# --------------------
np.random.seed(1)

# --------------------
# Load Data
# --------------------
A_train = np.load("A_train.npy")
b_train = np.load("b_train.npy")
A_test = np.load("A_test.npy")
b_test = np.load("b_test.npy")
x_start = np.zeros(A_train.shape[1])

# ---------------------------
# Run Descent Algos on LASSO
# ---------------------------

# Set up Descent Structure

T = int(1e4)
alpha, beta = utils.get_alpha_beta(A_train)
data = get_args_dict(('A', 'b'), (A_train, b_train))
test_data = utils.get_args_dict(('A', 'b'), (A_test, b_test))
params = get_args_dict(('beta', 'lam', 'T', 'c'), (beta, 10.0, T, 1e-5))
gd = descent_structure(data, params)


# LASSO using Proximal Gradient Descent
error_fn = utils.get_l2_loss
norm_fn = utils.get_l1_loss
subgrad_fn = utils.get_l2_subgrad
x_pgd, error_pgd, l1_pgd, xs_pgd = gd.descent(proximal_gradient_update, subgrad_fn, error_fn, norm_fn, x_start)
test_errors_pgd = test_error(xs_pgd, test_data, params)
norm_star1 = l1_pgd[-1]


# LASSO using subgradient method
x_sg, error_sg, l1_sg, xs_sg = gd.descent(subgradient_update, subgrad_fn, error_fn, norm_fn, x_start)
test_errors_sg = test_error(xs_sg, test_data, params)
norm_star2 = l1_sg[-1]
norm_star = (norm_star1 + norm_star2)/2.0


# LASSO using FW methods
gd.parameters['gamma'] = norm_star
x_fw, error_fw, l1_fw, xs_fw = gd.descent(frank_wolfe_update, subgrad_fn, error_fn, norm_fn, x_start)
test_errors_fw = test_error(xs_fw, test_data, params)



# ---------------------------
# Plots + Save
# ---------------------------
plt.clf()
plt.plot(error_sg, label='Subgradient')
plt.plot(error_fw, label='Frank-Wolfe')
plt.plot(error_pgd, label='Proximal GD')
plt.title('Training Error')
plt.legend()
plt.savefig('p1/train_error.pdf')

plt.clf()
plt.plot(l1_sg, label='Subgradient')
plt.plot(l1_fw, label='Frank-Wolfe')
plt.plot(l1_pgd, label='Proximal GD')
plt.title("$\ell^1$ Norm")
plt.legend()
plt.savefig('p1/l1.pdf')

plt.clf()
plt.plot(test_errors_sg, label='Subgradient')
plt.plot(test_errors_fw, label='Frank-Wolfe')
plt.plot(test_errors_pgd, label='Proximal GD')
plt.title("Test Error")
plt.legend()
plt.savefig('p1/test_error.pdf')
