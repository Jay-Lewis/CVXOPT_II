import utils
import numpy as np

# Load Data
# --------------------
m = 1000
n = 500
A = np.random.randn(1, n)
x_true = np.random.randn(n, 1)
x = np.random.randn(n, 1)
b = np.matmul(A, x_true).reshape([-1,])

# ---------------------------
# Test
# ---------------------------

# Set up Descent Structure
T = int(1e3)
alpha, beta = utils.get_alpha_beta(A)
data = utils.get_args_dict(('A', 'b'), (A, b))


utils.get_logist_subgrad(x, data, None)
