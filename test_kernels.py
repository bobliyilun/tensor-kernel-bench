import argparse
import unittest

from benchmark import tile_sizes
from kernels import (
    batched_matmul,
    matmul,
    matmul_bias,
    matmul_gelu,
    matmul_relu,
    matmul_transposed_right,
    max_abs_difference,
    tiled_matmul,
    tiled_matmul_transposed_right,
)


class KernelTests(unittest.TestCase):
    def test_matmul_relu_applies_activation_after_accumulation(self):
        self.assertEqual(
            matmul_relu([[1.0, 2.0], [-1.0, 1.0]], [[2.0, -1.0], [-1.0, 0.0]]),
            [[0.0, 0.0], [0.0, 1.0]],
        )

    def test_matmul_gelu_matches_exact_definition(self):
        actual = matmul_gelu([[1.0, -1.0]], [[1.0, 0.0], [0.0, 1.0]])
        self.assertAlmostEqual(actual[0][0], 0.8413447460685429)
        self.assertAlmostEqual(actual[0][1], -0.15865525393145707)

    def test_matmul_bias_fuses_column_bias_and_validates_shape(self):
        self.assertEqual(
            matmul_bias([[1.0, 2.0], [3.0, 4.0]], [[2.0, 1.0], [0.0, -1.0]], [0.5, -2.0]),
            [[2.5, -3.0], [6.5, -3.0]],
        )
        with self.assertRaises(ValueError):
            matmul_bias([[1.0]], [[2.0, 3.0]], [1.0])

    def test_batched_matmul_matches_individual_products(self):
        left = [[[1.0, 2.0], [3.0, 4.0]], [[-1.0, 0.0], [2.0, 1.0]]]
        right = [[[2.0], [1.0]], [[3.0], [-2.0]]]
        self.assertEqual(batched_matmul(left, right), [[[4.0], [10.0]], [[-3.0], [4.0]]])

    def test_batched_matmul_rejects_bad_batch_dimensions(self):
        with self.assertRaises(ValueError):
            batched_matmul([], [])
        with self.assertRaises(ValueError):
            batched_matmul([[[1.0]]], [])

    def test_tile_sizes_parses_and_rejects_invalid_values(self):
        self.assertEqual(tile_sizes("1,4,16"), [1, 4, 16])
        with self.assertRaises(argparse.ArgumentTypeError):
            tile_sizes("0,4")
        with self.assertRaises(argparse.ArgumentTypeError):
            tile_sizes("four")

    def test_tiled_matches_reference_for_rectangular_input(self):
        left = [[1.0, 2.0, 3.0], [-1.0, 0.0, 4.0]]
        right = [[2.0, 1.0], [0.0, -1.0], [3.0, 2.0]]
        expected = matmul(left, right)
        actual = tiled_matmul(left, right, tile=2)
        self.assertLessEqual(max_abs_difference(expected, actual), 1e-12)

    def test_rejects_bad_shapes_and_tiles(self):
        with self.assertRaises(ValueError):
            matmul([[1.0]], [[1.0, 2.0], [3.0, 4.0]])
        with self.assertRaises(ValueError):
            tiled_matmul([[1.0]], [[1.0]], tile=0)

    def test_transposed_right_layout_matches_reference_for_rectangular_input(self):
        left = [[1.0, 2.0, 3.0], [-1.0, 0.0, 4.0]]
        right = [[2.0, 1.0], [0.0, -1.0], [3.0, 2.0]]
        right_transposed = [[2.0, 0.0, 3.0], [1.0, -1.0, 2.0]]
        expected = matmul(left, right)
        self.assertLessEqual(
            max_abs_difference(expected, matmul_transposed_right(left, right_transposed)), 1e-12
        )
        self.assertLessEqual(
            max_abs_difference(
                expected, tiled_matmul_transposed_right(left, right_transposed, tile=2)
            ),
            1e-12,
        )

    def test_transposed_right_layout_rejects_bad_inner_dimension(self):
        with self.assertRaises(ValueError):
            matmul_transposed_right([[1.0, 2.0]], [[1.0]])


if __name__ == "__main__":
    unittest.main()
