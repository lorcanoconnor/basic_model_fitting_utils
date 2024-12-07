import numpy as np


def k_folds(n_samples, k, shuffle=True):
    ixs = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(ixs)
    return np.array_split(ixs, k)


def _get_jth_cross_val_split_ixs(folds, j):
    """
    0-indexed
    """
    val_ixs = folds[j]
    train_ixs = np.concatenate([fold for i, fold in enumerate(folds) if i != j])
    return train_ixs, val_ixs


def cross_validation(fn, fold_ixs, *extra_func_args):
    """
    fn: (train_ixs, val_ixs, *extra_func_args) -> metric
    e.g. train, val, ridge penalties -> MSE
    """
    res = []
    for j in range(len(fold_ixs)):
        train_ixs, val_ixs = _get_jth_cross_val_split_ixs(fold_ixs, j)
        metric = fn(train_ixs, val_ixs, *extra_func_args)
        res.append(metric)
    return res