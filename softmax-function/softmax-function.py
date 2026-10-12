import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    arr = np.array(x, dtype=float)
    if arr.ndim == 1:
        m = np.max(arr)
        shifted = arr - m
        exp_vals = np.exp(shifted)
        sum_vals = np.sum(exp_vals)
        return exp_vals / sum_vals
    max_vals = np.max(arr, axis=1, keepdims=True)
    shifted = arr - max_vals
    exp_vals = np.exp(shifted)
    sum_vals = np.sum(exp_vals, axis=1, keepdims=True)
    return exp_vals / sum_vals