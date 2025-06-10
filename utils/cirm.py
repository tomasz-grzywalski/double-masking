import numpy as np

K = 10.0
C = 0.1


def generate_mask(S_r, S_i, Y_r, Y_i):
    M_r = (Y_r * S_r + Y_i * S_i) / (Y_r ** 2.0 + Y_i ** 2.0 + 1e-5)
    M_i = (Y_r * S_i - Y_i * S_r) / (Y_r ** 2.0 + Y_i ** 2.0 + 1e-5)

    M_r = K * (1.0 - np.exp(-C * M_r)) / (1.0 + np.exp(-C * M_r))
    M_i = K * (1.0 - np.exp(-C * M_i)) / (1.0 + np.exp(-C * M_i))

    return M_r, M_i


def apply_mask(M_r, M_i, Y_r, Y_i):
    M_r = np.clip(M_r, -10.0 + 1e-6, 10.0 - 1e-6)
    M_i = np.clip(M_i, -10.0 + 1e-6, 10.0 - 1e-6)
    M_r = -(1.0 / C) * np.log((K - M_r) / (K + M_r))
    M_i = -(1.0 / C) * np.log((K - M_i) / (K + M_i))

    S_r = M_r * Y_r - M_i * Y_i
    S_i = M_r * Y_i + M_i * Y_r

    return S_r, S_i
