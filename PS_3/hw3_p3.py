import matplotlib.pyplot as plt
from descent_algos import *
import utils


# ==========================================
# HW 3 (Problem 3)
# ==========================================

# --------------------
# Fix Random Seed
# --------------------


# --------------------
# Load Data
# --------------------
m = 1000
n = 500
A = np.random.randn(m, n)
x_true = np.random.randn(n,)
b = np.matmul(A, x_true).reshape([-1,])

# ---------------------------
# Run Descent Algos on LASSO
# ---------------------------

# Set up Descent Structure
T = int(1e3)
alpha, beta = utils.get_alpha_beta(A)
data = utils.get_args_dict(('A', 'b'), (A, b))
parameters = utils.get_args_dict(('alpha', 'beta', 'lam', 'T', 'c'), (alpha, beta, 1.0, T, 1e-5))
gd = descent_structure(data, parameters)

fig1 = plt.figure()
ax1 = plt.axes()
fig2 = plt.figure()
ax2 = plt.axes()

for i in range(0, 3):
    # np.random.seed(i)
    # Logistic Regression using subgradient method
    x_random = np.random.randn
    subgrad_fn = utils.get_logist_subgrad
    x_sg, error_sg, l1_sg, xs_sg = gd.descent(subgradient_update, subgrad_fn, )
    test_errors_sg = utils.test_error(xs_sg, A, b)


    # Logistic Regression using accelerated subgradient method
    ab_tuple = (alpha, beta)
    x_F, error_F, l1_F, xs_F = gd.accelerated_descent(accelerated_subgrad_update, subgrad_fn)

    ax1.plot(error_sg, label='Subgrad_'+str(i))
    ax1.plot(error_F, label='Accel. Subgrad_'+str(i))
    ax2.plot(l1_sg, label='Subgradient_'+str(i))
    ax2.plot(l1_F, label='Accel. Subgrad_'+str(i))

# ---------------------------
# Save Plots
# ---------------------------

ax1.legend()
fig1.savefig('p3/train_error.eps')

ax2.legend()
fig2.savefig('p3/l1.eps')

