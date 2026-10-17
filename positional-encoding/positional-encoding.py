import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    pos = np.arange(seq_len)[:,np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = pos / np.power( base, (2 * (i // 2) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angle_rates[:, 0::2])
    pe[:, 1::2] = np.cos(angle_rates[:, 1::2])
    return pe