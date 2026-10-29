import numpy as np
from sklearn.metrics import balanced_accuracy_score


def correlate(X, Y, keep_dims=False):
    """correlation of two matrices, return corr per dimension

    Params:
    - keep_dims: return a list of corr per feature dim
    """

    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    X = X - X.mean(0)
    Y = Y - Y.mean(0)

    SX2 = (X**2).sum(0) ** 0.5
    SY2 = (Y**2).sum(0) ** 0.5
    SXY = (X * Y).sum(0)

    if not keep_dims:
        return np.mean(SXY / (SX2 * SY2 + 1e-8))
    else:
        return SXY / (SX2 * SY2 + 1e-8)


def balanced_accuracy(
    y_true: np.array, y_pred: np.array
):  # to match the syntax in the main script, not to mix up y_true and y_pred
    """wrapper function for calling balanced accuracy from sklearn"""
    return balanced_accuracy_score(y_pred=y_pred, y_true=y_true)


