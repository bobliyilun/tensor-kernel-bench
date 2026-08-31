"""Correctness-first tensor kernel references."""

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


def max_abs_difference(left: Matrix, right: Matrix) -> float:
    if _shape(left) != _shape(right):
        raise ValueError("matrix shapes must match")
    return max(abs(a - b) for row_a, row_b in zip(left, right) for a, b in zip(row_a, row_b))

