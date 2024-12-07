import numpy as np
import cvxpy as cp
from matplotlib import pyplot as plt

def fit_isotropic_ridge__cvxpy(X, y, penalty_arr, fit_intercept=True):
    """
    It is silly to use cvxpy for ridge OLS. But can copy from this template to do LASSO etc.
    NOTE: unconventional that penalty_arr is on scale of beta, not of beta^2 (IMO more intuitive to correspond to
    standard dev in the bayesian normal prior MAP interpretation)
    """
    p = X.shape[1]
    assert len(penalty_arr) == p
    beta = cp.Variable(p)
    if fit_intercept:
        intercept = cp.Variable(1)
    else:
        intercept = cp.Constant(0)
    loss = cp.sum_squares(y - X @ beta - intercept) + cp.pnorm(cp.multiply(penalty_arr, beta), p=2)**2
    problem = cp.Problem(cp.Minimize(loss))
    problem.solve()
    if fit_intercept:
        return beta.value, intercept.value
    else:
        return beta.value, None

def fit_isotropic_ridge(X, y, penalty_arr, fit_intercept=True):
    """
    Analytic version.
    NOTE: unconventional that penalty_arr is on scale of beta, not of beta^2 (IMO more intuitive to correspond to
    standard dev in the bayesian normal prior MAP interpretation)
    """
    n, p = X.shape
    c = np.concatenate
    assert len(penalty_arr) == p
    if not fit_intercept:
        _X, _y = c([X, np.diag(np.sqrt(penalty_arr))], axis=0), c([y, np.zeros(p)], axis=0)
        beta, intercept = fit_ols(_X, _y), None
        return beta, intercept
    _intercept = np.ones((n, 1))
    full_sol, _ = fit_isotropic_ridge(
        c([_intercept, X], axis=1), y, penalty_arr=c([np.array([0]), penalty_arr]), fit_intercept=False
    )
    intercept, beta = full_sol[0], full_sol[1:]
    return beta, intercept


def fit_ols(X, y):
    return np.linalg.solve(X.T.dot(X), X.T.dot(y))


if __name__ == '__main__':
    ### test case
    np.random.seed(1)
    N = 10_000
    true_beta = [5, 7]
    true_intercept = 3
    X = np.random.rand(N, 2)
    y = true_intercept + X.dot(true_beta) + np.random.standard_normal(size=N)

    beta, intercept = fit_isotropic_ridge(X, y, np.array([0, 0]), fit_intercept=True)
    print(f'{beta=}')
    print(f'{intercept=}')