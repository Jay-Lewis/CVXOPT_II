from ProblemSets import utils


def p3_error(Xs_gd, M, data, params):
    diffs = [X_i - M for X_i in Xs_gd]
    errors = utils.test_error(diffs, utils.frobenius_norm, data, params)
    return [utils.to_numpy(error) ** 2 for error in errors]

def p4_get_steps(s, T, N):
    xs = []
    m = 0
    for i in range(1, T+1):
        m = m +1
        if i % s == 0:
            m = m + N
        xs.append(m)

    return xs
