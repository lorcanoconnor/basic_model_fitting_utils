import numpy as np
from matplotlib import pyplot as plt


def plot_std_res_vs_influence(ols_res, cooks_thresh=np.nan, **plot_kwargs):
    """
    standard cooks rule of thumb is 8 / (n - 2p)
    """
    inf = ols_res.get_influence()
    cooks, _ = inf.cooks_distance
    cols = ['r' if c > cooks_thresh else 'b' for c in cooks]
    plt.scatter(
    inf.hat_matrix_diag, inf.resid / inf.resid_std, marker='.', alpha=.1,
        c=cols, **plot_kwargs
)
    plt.ylabel('Standarised residual')
    plt.xlabel('Influence (i.e. diag of H)')

    plt.grid()
    plt.show()


def plot_resid_vs_fitted(ols_res, **plot_kwargs):
    plt.scatter(ols_res.fittedvalues, ols_res.resid, marker='.', alpha=.1, **plot_kwargs)
    plt.ylabel('Fitted')
    plt.xlabel('Residual')

    plt.grid()
    plt.show()

def plot_scale_vs_location(ols_res, **plot_kwargs):
    inf = ols_res.get_influence()
    plt.scatter(ols_res.fittedvalues, np.sqrt(np.abs(inf.resid  / inf.resid_std)), marker='.', alpha=.1, **plot_kwargs)
    plt.ylabel('Fitted')
    plt.xlabel('sqrt(standardised residual)')

    plt.grid()
    plt.show()