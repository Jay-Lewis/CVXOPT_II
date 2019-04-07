# ===========
# Imports
# ===========

import numpy as np
import matplotlib.pyplot as plt

##############################################################################
#                                                                            #
#                               K_TILDE TESTS                                #
#                                                                            #
##############################################################################


# # =======================
# # Test 0
# # =======================
#
# # investigate concentration of emp. cov. to exp. cov matrix as function of dimension
#
# # ------ Loop dimension -----
# dims = [int(d) for d in np.linspace(2, 100, 4)]
# steps = []
#
# for d in dims:
#     diffs = []
#     n = 100*d
#
#     secnd_moments = np.random.randn(d)**2
#     cov = np.diag(secnd_moments)
#     mu = np.zeros(d)
#
#     exp_norm = np.sum(secnd_moments)
#     # rhs = exp_norm*cov
#     rhs = cov
#     estimate = np.zeros([d, d])
#     nums = range(1, n+1)
#     pts = []
#
#     # print('Covariance Matrix: ', cov)
#
#     # ------ Estimate Cov. Matrix -----
#     step = None
#     for current_num in nums:
#         a = np.random.multivariate_normal(mu, cov)
#         pts.append(a)
#         norm = np.linalg.norm(a)
#         matrix = np.outer(a.T, a.T)
#         # estimate = estimate + norm*matrix
#         estimate = estimate + matrix
#         diff = np.linalg.norm(estimate/current_num - rhs)
#         diffs.append(diff)
#         if diff < 1:
#             step = current_num
#     if step is None:
#         step = current_num
#         print('THIS HAPPENED')
#         print(diff)
#     steps.append(step)
#
#
#     # print('--------------')
#     # print('Estimate: ')
#     # print(estimate/current_num)
#     # print('N: ', current_num)
#     # print('final diff:', diff)
#
# # ------ Plot concentration -----
#
# plt.plot(dims, steps)
# plt.show()
#
# # ------ END Loop dimension -----







# =======================
# Test 1
# =======================

# verify that E[norm(a)a*a^T] ---> E[norm(a)]*E[a*a^T]

diffs = []
d = 10
n = 500*d

secnd_moments = np.random.randn(d)**2
cov = np.diag(secnd_moments)
mu = np.zeros(d)

exp_norm = np.sum(secnd_moments)
# rhs = exp_norm*cov
rhs = cov
estimate = np.zeros([d, d])
nums = range(1, n+1)
pts = []

print('Covariance Matrix: ', cov)

for current_num in nums:
    a = np.random.multivariate_normal(mu, cov)
    pts.append(a)
    norm = np.linalg.norm(a)
    matrix = np.outer(a.T, a.T)
    # estimate = estimate + norm*matrix
    estimate = estimate + matrix
    diff = np.linalg.norm(estimate/current_num - rhs)
    diffs.append(diff)
print('--------------')
print('Estimate: ')
print(estimate/current_num)
print('N: ', current_num)
print('final diff:', diff)



plt.plot(nums, diffs)
plt.show()





# =======================
# xxxxxxx
# =======================