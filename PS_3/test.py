import utils
import numpy as np

# Load Data
# --------------------
m = 2
n = 10
A = np.random.randn(m, n)
x_true = np.random.randn(n,)
x = np.random.randn(n,)
b = np.matmul(A, x_true).reshape([-1,])

# ---------------------------
# Test
# ---------------------------

# Set up Descent Structure
T = int(1e3)
alpha, beta = utils.get_alpha_beta(A)
data = utils.get_args_dict(('A', 'b'), (A, b))
params = {'lam': 1.0}


utils.get_logist_subgrad(x, data, None)
grad = utils.get_LASSO_subgrad(x, data, params)
print(grad)
