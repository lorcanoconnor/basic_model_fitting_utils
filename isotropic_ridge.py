import numpy as np
import cvxpy as cp
from matplotlib import pyplot as plt
from tqdm import tqdm

from cross_validation import cross_validation, k_folds


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
    intercept, beta = full_sol[0].item(), full_sol[1:]
    return beta, intercept


def fit_ols(X, y):
    return np.linalg.solve(X.T.dot(X), X.T.dot(y))


def mse(pred, obs):
    return np.mean((pred - obs) ** 2)


def ridge_cv_mse(penalties, fold_ixs):
    def fn(train_ixs, val_ixs):
        X_train, X_val = X[train_ixs], X[val_ixs]
        y_train, y_val = y[train_ixs], y[val_ixs]
        beta, intercept = fit_isotropic_ridge(X_train, y_train, penalties, fit_intercept=True)
        pred = X_val.dot(beta) + intercept
        return mse(pred, y_val)

    res = cross_validation(fn, fold_ixs)
    return np.mean(res)

#TODO: optimise for groups of ridge penalties, e.g. penalties=[lam1, lam1, lam2]

if __name__ == '__main__':
    ### example
    N, p = 10_000, 50
    sig, tau = 10, 0.0001
    true_beta = np.random.standard_normal(size=p) * tau
    true_intercept = 3
    X = np.random.rand(N, p)
    y = true_intercept + X.dot(true_beta) + np.random.standard_normal(size=N) * sig

    # beta, intercept = fit_isotropic_ridge(X, y, np.array([10] * p), fit_intercept=True)
    # print(f'{beta=}')
    # print(f'{intercept=}')

    fold_ixs = k_folds(len(X), k=5, shuffle=True)
    signal_noise_guess = sig / (tau * np.sqrt(N))
    lams = np.geomspace(signal_noise_guess / 100, signal_noise_guess * 100, 10)
    res = []
    for lam in tqdm(lams):
        res.append(ridge_cv_mse(penalties=np.array([lam] * p), fold_ixs=fold_ixs))
    plt.plot(lams, res)
    plt.grid()
    plt.xscale('log')
    plt.axvline(signal_noise_guess, linestyle='--', label='SNR guess', c='k')
    plt.legend()
    plt.show()
