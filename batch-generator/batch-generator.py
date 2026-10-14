import numpy as np

def batch_generator(X: list, y: list, batch_size: int, seed: int = 42, drop_last: bool = False):
    """
    Returns a generator of (X_batch, y_batch) tuples.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    n_samples = len(X_arr)
    rng = np.random.default_rng(seed)
    indices= np.arange(n_samples)
    rng.shuffle(indices)

    for start_index in range(0, n_samples, batch_size):
        end_idx = start_index + batch_size
        if end_idx > n_samples:
            if drop_last:
                break
            else:
                end_idx = n_samples

        batch_indices = indices[start_index: end_idx]
        X_batch = X_arr[batch_indices]
        y_batch = y_arr[batch_indices]

        yield (X_batch, y_batch)