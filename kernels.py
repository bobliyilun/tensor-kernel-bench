"""Correctness-first tensor kernel references."""

import math
from typing import List, Sequence, Tuple

Matrix = List[List[float]]


def _shape(matrix: Sequence[Sequence[float]]) -> Tuple[int, int]:
    if not matrix or not matrix[0]:
        raise ValueError("matrices must not be empty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrices must be rectangular")
    return len(matrix), width


def matmul(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> Matrix:
    m, k = _shape(left)
    right_k, n = _shape(right)
    if k != right_k:
        raise ValueError("inner dimensions must match")
    output = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for p in range(k):
            value = left[i][p]
            for j in range(n):
                output[i][j] += value * right[p][j]
    return output


def matmul_bias(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]], bias: Sequence[float]
) -> Matrix:
    """Multiply matrices and add a column-wise bias in the output loop."""
    m, k = _shape(left)
    right_k, n = _shape(right)
    if k != right_k:
        raise ValueError("inner dimensions must match")
    if len(bias) != n:
        raise ValueError("bias dimension must match output columns")
    output = [list(bias) for _ in range(m)]
    for i in range(m):
        for p in range(k):
            value = left[i][p]
            for j in range(n):
                output[i][j] += value * right[p][j]
    return output


def matmul_relu(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> Matrix:
    """Multiply matrices and apply a ReLU epilogue in the output loop."""
    m, k = _shape(left)
    right_k, n = _shape(right)
    if k != right_k:
        raise ValueError("inner dimensions must match")
    output = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for p in range(k):
            value = left[i][p]
            for j in range(n):
                output[i][j] += value * right[p][j]
        for j in range(n):
            output[i][j] = max(0.0, output[i][j])
    return output


def matmul_gelu(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> Matrix:
    """Multiply matrices and apply the exact GELU epilogue in the output loop."""
    m, k = _shape(left)
    right_k, n = _shape(right)
    if k != right_k:
        raise ValueError("inner dimensions must match")
    output = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for p in range(k):
            value = left[i][p]
            for j in range(n):
                output[i][j] += value * right[p][j]
        for j in range(n):
            value = output[i][j]
            output[i][j] = 0.5 * value * (1.0 + math.erf(value / math.sqrt(2.0)))
    return output


def batched_matmul(
    left: Sequence[Sequence[Sequence[float]]], right: Sequence[Sequence[Sequence[float]]]
) -> List[Matrix]:
    """Multiply matching batches of matrices."""
    if not left or not right:
        raise ValueError("batches must not be empty")
    if len(left) != len(right):
        raise ValueError("batch dimensions must match")
    return [matmul(left_matrix, right_matrix) for left_matrix, right_matrix in zip(left, right)]


def matmul_transposed_right(
    left: Sequence[Sequence[float]], right_transposed: Sequence[Sequence[float]]
) -> Matrix:
    """Multiply ``left`` by a right-hand side stored as ``[n][k]``."""
    m, k = _shape(left)
    n, right_k = _shape(right_transposed)
    if k != right_k:
        raise ValueError("inner dimensions must match")
    return [
        [sum(left[i][p] * right_transposed[j][p] for p in range(k)) for j in range(n)]
        for i in range(m)
    ]


def tiled_matmul(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]], tile: int = 16
) -> Matrix:
    m, k = _shape(left)
    right_k, n = _shape(right)
    if k != right_k:
        raise ValueError("inner dimensions must match")
    if tile <= 0:
        raise ValueError("tile must be positive")
    output = [[0.0] * n for _ in range(m)]
    for ii in range(0, m, tile):
        for pp in range(0, k, tile):
            for jj in range(0, n, tile):
                for i in range(ii, min(ii + tile, m)):
                    for p in range(pp, min(pp + tile, k)):
                        value = left[i][p]
                        for j in range(jj, min(jj + tile, n)):
                            output[i][j] += value * right[p][j]
    return output


def tiled_matmul_transposed_right(
    left: Sequence[Sequence[float]], right_transposed: Sequence[Sequence[float]], tile: int = 16
) -> Matrix:
    """Tiled multiplication with a right-hand side stored as ``[n][k]``."""
    m, k = _shape(left)
    n, right_k = _shape(right_transposed)
    if k != right_k:
        raise ValueError("inner dimensions must match")
    if tile <= 0:
        raise ValueError("tile must be positive")
    output = [[0.0] * n for _ in range(m)]
    for ii in range(0, m, tile):
        for jj in range(0, n, tile):
            for pp in range(0, k, tile):
                for i in range(ii, min(ii + tile, m)):
                    for j in range(jj, min(jj + tile, n)):
                        for p in range(pp, min(pp + tile, k)):
                            output[i][j] += left[i][p] * right_transposed[j][p]
    return output


def max_abs_difference(left: Matrix, right: Matrix) -> float:
    if _shape(left) != _shape(right):
        raise ValueError("matrix shapes must match")
    return max(abs(a - b) for row_a, row_b in zip(left, right) for a, b in zip(row_a, row_b))
