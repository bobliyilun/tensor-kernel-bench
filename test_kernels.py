import unittest

from kernels import (
    matmul,
    matmul_transposed_right,
    max_abs_difference,
    tiled_matmul,
    tiled_matmul_transposed_right,
)


class KernelTests(unittest.TestCase):
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
