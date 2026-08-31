import unittest

from kernels import matmul, max_abs_difference, tiled_matmul


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


if __name__ == "__main__":
    unittest.main()

