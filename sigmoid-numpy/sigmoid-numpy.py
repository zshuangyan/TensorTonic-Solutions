import numpy as np

def pure_numpy(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    arr = np.asarray(x, dtype=float)
    res = pure_numpy(arr)

    if np.ndim(x) == 0:
        return float(res)

    return res