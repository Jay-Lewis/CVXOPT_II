
def fuzzy_equal(a, b, tol=1e-8):
    diff = a-b
    return diff <= tol
