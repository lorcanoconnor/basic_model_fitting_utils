import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ydata_profiling import ProfileReport

def impute_random_categorical(X: pd.Series, allowed_vals=None):
    allowed_vals = allowed_vals or X.unique()
    nice_mask = X.isin(allowed_vals)
    if (~nice_mask).sum() == 0:
        print('No imputation needed')
        return X.copy()
    output = X.copy()
    output.loc[~nice_mask] = np.random.choice(X.loc[nice_mask], (~nice_mask).sum())
    return output


def impute_random_cts(X: pd.Series, allowed_min=-np.inf, allowed_max=np.inf):
    nice_mask = (X >= allowed_min) & (X <= allowed_max)  # excludes np.nan deliberately
    if (~nice_mask).sum() == 0:
        print('No imputation needed')
        return X.copy()
    output = X.copy()
    output.loc[~nice_mask] = np.random.choice(X.loc[nice_mask], (~nice_mask).sum())
    return output


def winsorise(X: pd.Series, quantile=.05):
    l, h = X.quantile(quantile), X.quantile(1 - quantile)
    return X.clip(l, h)


def print_chunks(lst, chunk_size=10):
    for i in range(0, len(lst), chunk_size):
        print(lst[i:i + chunk_size])


def simple_non_shuffled_test_train_split(df):
    df_train, df_test = train_test_split(df, train_size=.8, shuffle=False)
    return df_train, df_test


def vcs(data, col=None):
    return data[col].value_counts(normalize=False, dropna=True)

def get_eda_report():
    profile = ProfileReport(df, title="EDA Report", explorative=True)
    profile.to_file("eda_report.html")
