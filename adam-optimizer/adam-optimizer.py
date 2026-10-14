import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    m_arr = np.asarray(m)
    v_arr = np.asarray(v)
    g_arr = np.asarray(grad)
    param_arr = np.asarray(param)
    
    # 修正 Beta1 为小写 beta1
    m_next = beta1 * m_arr + (1 - beta1) * g_arr
    v_next = beta2 * v_arr + (1 - beta2) * (g_arr ** 2)
    
    m_upper_next = m_next / (1 - beta1 ** t)
    v_upper_next = v_next / (1 - beta2 ** t)
    
    # 使用 param_arr 代替 param
    param_next = param_arr - lr * m_upper_next / (v_upper_next ** (1 / 2) + eps)
    
    return (param_next, m_next, v_next)