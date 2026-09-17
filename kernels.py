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


def matmul_numpy(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> Matrix:
    """Multiply matrices through NumPy when its optional dependency is installed."""
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("NumPy backend requires numpy; install it with pip") from error
    return np.matmul(np.asarray(left), np.asarray(right)).tolist()


def matmul_torch(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> Matrix:
    """Multiply matrices through eager PyTorch when its optional dependency is installed."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch backend requires torch; install it with pip") from error
    return torch.matmul(torch.tensor(left), torch.tensor(right)).tolist()


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


def row_softmax(values: Sequence[Sequence[float]]) -> Matrix:
    """Apply the softmax function independently to every row."""
    _shape(values)
    output = []
    for row in values:
        exponentials = [math.exp(value) for value in row]
        total = sum(exponentials)
        output.append([value / total for value in exponentials])
    return output


def row_softmax_online(values: Sequence[Sequence[float]]) -> Matrix:
    """Apply numerically stable softmax to each row using an online maximum."""
    _shape(values)
    output = []
    for row in values:
        maximum = -math.inf
        total = 0.0
        for value in row:
            if value <= maximum:
                total += math.exp(value - maximum)
            else:
                total = total * math.exp(maximum - value) + 1.0
                maximum = value
        output.append([math.exp(value - maximum) / total for value in row])
    return output


def row_layer_norm(values: Sequence[Sequence[float]], epsilon: float = 1e-5) -> Matrix:
    """Normalize each row to zero mean and unit variance."""
    _shape(values)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    output = []
    for row in values:
        mean = sum(row) / len(row)
        variance = sum((value - mean) ** 2 for value in row) / len(row)
        scale = math.sqrt(variance + epsilon)
        output.append([(value - mean) / scale for value in row])
    return output


def causal_attention(
    query: Sequence[Sequence[float]],
    key: Sequence[Sequence[float]],
    value: Sequence[Sequence[float]],
) -> Matrix:
    """Apply scaled dot-product attention, masking positions after each query."""
    sequence_length, head_size = _shape(query)
    key_length, key_size = _shape(key)
    value_length, value_size = _shape(value)
    if key_length != sequence_length or value_length != sequence_length:
        raise ValueError("query, key, and value sequence lengths must match")
    if key_size != head_size:
        raise ValueError("query and key head dimensions must match")

    scale = 1.0 / math.sqrt(head_size)
    output = []
    for position, query_row in enumerate(query):
        scores = [
            sum(q * k for q, k in zip(query_row, key[index])) * scale
            for index in range(position + 1)
        ]
        maximum = max(scores)
        weights = [math.exp(score - maximum) for score in scores]
        total = sum(weights)
        output.append(
            [
                sum(weights[index] * value[index][column] for index in range(position + 1))
                / total
                for column in range(value_size)
            ]
        )
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
