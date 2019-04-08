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
# # Test -1
# # =======================
#
# # LLN for gaussian random variable
#
# estimates = []
# ns = [_ for _ in range(10, 1000)]
# pts = np.asarray([])
# for n in ns:
#     xs = np.random.randn(n)
#     pts = np.hstack((pts, xs))
#     estimate = np.sum(xs**2)/n
#     estimates.append(estimate)
#
# plt.plot(ns, estimates)
# plt.show()

# # =======================
# # Test 0
# # =======================
#
# # investigate concentration of emp. cov. to exp. cov matrix as function of dimension
#
# # ------ Loop dimension -----
# dims = [int(d) for d in np.linspace(2, 100, 2)]
#
# for d in dims:
#     diffs = []
#     n = 1000
#     secnd_moments = np.random.randn(d)**2
#     cov = np.diag(secnd_moments)
#     mu = np.zeros(d)
#     rhs = cov
#     estimate = np.zeros([d, d])
#     nums = range(1, n+1)
#
#     # ------ Estimate Cov. Matrix -----
#     for current_num in nums:
#         a = np.random.multivariate_normal(mu, cov)
#         norm = np.linalg.norm(a)
#         matrix = np.outer(a.T, a.T)
#         estimate = estimate + matrix
#         num_elements = d**2
#         avg_diff = np.linalg.norm(estimate/current_num - rhs)/num_elements
#         diffs.append(avg_diff)
#
#     # ------ Plot concentration -----
#
#     plt.plot(nums, diffs, label=str(d))
#     plt.ylabel('avg diff')
#     plt.xlabel('num samples')
#     plt.legend()
#
#     print(np.diagonal(cov)[0:20])
#     print('----------------------------------')
#     print(np.diagonal(estimate/current_num)[0:20])
#
# # ------ END Loop dimension -----
#
# plt.show()

# # =======================
# # Test 1a
# # =======================
#
# # verify that E[norm(a)^2 a*a^T] ---> E[norm(a)]*E[a*a^T]
# dims = [int(d) for d in np.linspace(2, 100, 4)]
# for d in dims:
#     diffs = []
#     n = 1000
#     secnd_moments = np.random.randn(d)**2
#     cov = np.diag(secnd_moments)
#     mu = np.zeros(d)
#     exp_norm = np.sum(secnd_moments)
#     rhs = exp_norm*cov
#     estimate = np.zeros([d, d])
#     nums = range(1, n+1)
#
#     print('True Matrix: ', np.diagonal(rhs))
#
#     for current_num in nums:
#         a = np.random.multivariate_normal(mu, cov)
#         norm = np.inner(a, a)
#         matrix = np.outer(a, a)
#         estimate = estimate + norm*matrix
#         diff = np.linalg.norm(estimate/current_num - rhs)
#         diffs.append(diff)
#
#     print('--------------')
#     print('Estimate: ')
#     print(np.diagonal(estimate/current_num))
#     print('N: ', current_num)
#     print('final diff:', diff)
#
#
#     plt.plot(nums, diffs, label='E[norm a*a^T] :' + str(d))
#
#
#     # =======================
#     # Test 1b
#     # =======================
#
#     # verify that E[a*a^T] ---> E[norm(a)]*E[a*a^T]
#
#     diffs = []
#     mu = np.zeros(d)
#     rhs = cov
#     estimate = np.zeros([d, d])
#     nums = range(1, n+1)
#
#     print('Cov Matrix: ', np.diagonal(rhs))
#
#     for current_num in nums:
#         a = np.random.multivariate_normal(mu, cov)
#         matrix = np.outer(a, a)
#         estimate = estimate + matrix
#         diff = np.linalg.norm(estimate/current_num - rhs)
#         diffs.append(diff)
#
#     print('--------------')
#     print('Estimate: ')
#     print(np.diagonal(estimate/current_num))
#     print('N: ', current_num)
#     print('final diff:', diff)
#
#     plt.plot(nums, diffs, label='E[a*a^T]' + str(d))
#
# plt.legend()
# plt.show()



# # =======================
# # Test 2a
# # =======================
#
# # verify that E[a*a^T/d]_{H^-1} is 1 (and highly concentrated for large d)
#
# d = 200
# diffs = []
# n = d
# secnd_moments = np.random.randn(d)**2
# cov = np.diag(secnd_moments)
# mu = np.zeros(d)
# nums = range(1, n+1)
# estimate = 0
# norms = []
#
# print('True value: ', 1)
#
# for current_num in nums:
#     a = np.random.multivariate_normal(mu, cov)
#     norm = np.matmul(a, np.matmul(np.linalg.inv(cov), a))
#     norms.append(norm/d)
#     estimate = estimate + norm/d
#     diff = np.abs(estimate/current_num - 1)
#     diffs.append(diff)
#
# print('--------------')
# print('Estimate: ')
# print(estimate/current_num)
# print('N: ', current_num)
# print('final diff:', diff)
#
# plt.plot(nums, diffs)
# plt.show()
#
# plt.hist(norms)
# plt.show()



# =======================
# Test 2b
# =======================

# verify that E[||a*a^T||_{H^-1}/d a*a^T] ---> H

d = 100
diffs = []
n = 1000
secnd_moments = np.random.randn(d)**2
cov = np.diag(secnd_moments)
mu = np.zeros(d)
exp_norm = np.sum(secnd_moments)
rhs = cov
estimate = np.zeros([d, d])
nums = range(1, n+1)

print('True Matrix: ', np.diagonal(rhs))

for current_num in nums:
    a = np.random.multivariate_normal(mu, cov)
    norm = np.matmul(a, np.matmul(np.linalg.inv(cov), a))
    matrix = np.outer(a, a)
    estimate = estimate + norm*matrix/d
    diff = np.linalg.norm(estimate/current_num - rhs)
    diffs.append(diff)

print('--------------')
print('Estimate: ')
print(np.diagonal(estimate/current_num))
print('N: ', current_num)
print('final diff:', diff)


plt.plot(nums, diffs, label='E[norm a*a^T] :' + str(d))
plt.show()