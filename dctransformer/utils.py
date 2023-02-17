import numpy as np
import torch
import torch.nn.functional as F



pi = torch.tensor(np.pi)

# padding_id = 0
# sos_id = 1
# eos_id = 2


# Base matrix for luma quantization
T_luma = torch.tensor([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99]
], dtype=torch.float)


# Chroma quantization matrix
T_chroma = torch.tensor([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
], dtype=torch.float)


def Q_luma(q=50, N=8):
    sq = 5000 / q if q < 50 else 200 - 2 * q
    t_luma = torch.floor(sq * T_luma / 100 + 0.5)
    return F.interpolate(t_luma[None, None, :, :], size=N, mode="nearest").squeeze()


def Q_chroma(N=8):
    return F.interpolate(T_chroma[None, None, :, :], size=N, mode="nearest").squeeze()


def zigzag(n: int) -> torch.Tensor:
    """Generates a zigzag position encoding tensor. 
    Source: https://github.com/benjs/DCTransformer-PyTorch/blob/main/dctransformer/transforms.py
    """

    pattern = torch.zeros(n, n)
    triangle = lambda x: (x * (x + 1)) / 2

    # even index sums
    for y in range(0, n):
        for x in range(y % 2, n - y, 2):
            pattern[y, x] = triangle(x + y + 1) - x - 1

    # odd index sums
    for y in range(0, n):
        for x in range((y + 1) % 2, n - y, 2):
            pattern[y, x] = triangle(x + y + 1) - y - 1

    # bottom right triangle
    for y in range(n - 1, -1, -1):
        for x in range(n - 1, -1 + (n - y), -1):
            pattern[y, x] = n * n - 1 - pattern[n-y-1, n-x-1]

    return pattern.t().contiguous()

