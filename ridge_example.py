import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import seaborn as sns
from cross_validation import k_folds, cross_validation
from isotropic_ridge import fit_isotropic_ridge, mse


def make_example_dataset():
    N, p_numeric = 10_000, 3
    cat_levels = (2, 10)
    df = pd.DataFrame(index=np.arange(N), dtype=int)
    df.loc[:, [f'num_{i}' for i in range(p_numeric)]] = np.random.rand(N, p_numeric)
    for i, q in enumerate(cat_levels):
        df.loc[:, f'cat_{i}'] = np.random.choice([f'level_{j}' for j in range(q)], N, replace=True)

    y = np.random.standard_normal(len(df)) + df['num_1'] + df.iloc[:, p_numeric:].apply(
        lambda x: sum((y.endswith('0') - y.endswith('1') for y in x)), axis=1)
    return df, y




def lams_to_penalties(lam_num, lam_cat, numeric_cols_in_dm, non_numeric_cols_in_dm):
    return np.concatenate([np.repeat(lam_num, numeric_cols_in_dm), np.repeat(lam_cat, non_numeric_cols_in_dm)])


if __name__ == '__main__':
    """
    Note: the CV done below is cheaty: preprocessing is done on whole dataset for simplicity. It should really be
    done afresh on each training fold. 
    """

    df, y = make_example_dataset()

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude='number').columns.tolist()

    dms = []
    colnames = []

    #### Process numeric cols (Bear in mind might not always follow this flow, e.g. might want is_missing cols derived
    #### from numeric cols.

    ss = StandardScaler()
    dms.append(ss.fit_transform(df[numeric_cols]))
    colnames.extend(ss.get_feature_names_out())
    numeric_cols_in_dm = len(ss.get_feature_names_out())

    # Process categoricals
    oh = OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore')
    dms.append(oh.fit_transform(df[non_numeric_cols]))
    colnames.extend(oh.get_feature_names_out())
    non_numeric_cols_in_dm= len(oh.get_feature_names_out())
    dm = np.concatenate(dms, axis=1)


    fold_ixs = k_folds(len(dm), k=5, shuffle=True)


    def ridge_cross_val(lam1, lam2, fold_ixs):
        penalties = lams_to_penalties(lam1, lam2, numeric_cols_in_dm, non_numeric_cols_in_dm)
        def fn(train_ixs, val_ixs):
            X_train, X_val = dm[train_ixs], dm[val_ixs]
            y_train, y_val = y[train_ixs], y[val_ixs]
            beta, intercept = fit_isotropic_ridge(X_train, y_train, penalties, fit_intercept=True)
            pred = X_val.dot(beta) + intercept
            return mse(pred, y_val)

        res = cross_validation(fn, fold_ixs)
        return np.mean(res)


    #axes orders and directions are confusing. Copy-paste this example for something that works

    lam1s = np.geomspace(10, 200, 20)
    lam2s = np.geomspace(1, 10, 20)

    res = np.zeros((len(lam1s), len(lam2s)))
    for i, lam1 in enumerate(lam1s):
        for j, lam2 in enumerate(lam2s):
            res[i, j] = ridge_cross_val(lam1, lam2, fold_ixs)
    best_i, best_j = np.unravel_index(res.argmin(), res.shape)


    ax = sns.heatmap(res, cmap='coolwarm', yticklabels=lam1s, xticklabels=lam2s,
                     vmin = res.mean() - 2*res.std(), vmax =res.mean() + 2*res.std()
                     )
    ax.text(best_j, best_i, '*', color='black', ha='center', va='center', fontsize=14)
    ax.invert_yaxis()
    plt.show()

    best_lams = lam1s[best_i], lam2s[best_j]
    print( lams_to_penalties(*best_lams, numeric_cols_in_dm, non_numeric_cols_in_dm))
    b, intercept = fit_isotropic_ridge(dm, y, lams_to_penalties(*best_lams, numeric_cols_in_dm, non_numeric_cols_in_dm), fit_intercept=True)
    print(pd.Series(b, index=colnames))
    print(f'{intercept=}')
