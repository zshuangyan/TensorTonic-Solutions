import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if not seqs:
        return np.empty((0,0) , dtype=int)
    num_of_sequences = len(seqs)
    max_length = max_len if max_len else max(len(seq) for seq in seqs)
    res_arr = np.full((num_of_sequences, max_length), pad_value, dtype=int)
    for i in range(num_of_sequences):
        seq_len = min(len(seqs[i]), max_length)
        res_arr[i, :seq_len] = seqs[i][:seq_len]
    return res_arr