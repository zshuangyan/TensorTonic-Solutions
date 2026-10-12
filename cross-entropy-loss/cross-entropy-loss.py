import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Returns the mean multiclass cross-entropy loss as a Python float.
    """
    np_pred = np.asarray(y_pred)
    num_of_samples = np_pred.shape[0]
    rows = np.arange(num_of_samples)
    cols = y_true
    np_sum = np.sum(np_pred, axis=1, keepdims=True)
    np_pb = np_pred / np_sum
    np_res = np_pb[rows, cols]
    np_total = np.sum(-np.log(np_res))
    return np_total / num_of_samples