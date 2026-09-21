import argparse
import csv
import math
import tempfile
import types
import unittest
from unittest.mock import patch

import kernels
from benchmark import load_thresholds, matmul_estimates, threshold_violations, tile_sizes, write_csv
from kernels import (
    batched_matmul,
    causal_attention,
    matmul,
    matmul_bias,
    matmul_gelu,
    matmul_numpy,
    matmul_relu,
    matmul_torch,
    matmul_torch_compile,
    matmul_triton,
    matmul_transposed_right,
    max_abs_difference,
    row_layer_norm,
    row_softmax,
    row_softmax_online,
    tiled_matmul,
    tiled_matmul_transposed_right,
)


class KernelTests(unittest.TestCase):
    def test_triton_backend_reports_missing_optional_dependency(self):
        with patch.dict("sys.modules", {"triton": None}):
            with self.assertRaisesRegex(RuntimeError, "requires torch and triton"):
                matmul_triton([[1.0]], [[2.0]])

    def test_torch_compile_backend_requires_supported_torch(self):
        with patch.dict("sys.modules", {"torch": types.SimpleNamespace()}), patch.object(
            kernels, "_compiled_torch_matmul", None
        ):
            with self.assertRaisesRegex(RuntimeError, "PyTorch 2.0"):
                matmul_torch_compile([[1.0]], [[2.0]])

    def test_torch_compile_backend_compiles_once(self):
        compiled = []

        def compile_matmul(function):
            compiled.append(function)
            return lambda left, right: types.SimpleNamespace(tolist=lambda: [[left[0][0] * right[0][0]]])

        torch = types.SimpleNamespace(
            compile=compile_matmul, matmul=object(), tensor=lambda values: values
        )
        with patch.dict("sys.modules", {"torch": torch}), patch.object(
            kernels, "_compiled_torch_matmul", None
        ):
            self.assertEqual(matmul_torch_compile([[3.0]], [[2.0]]), [[6.0]])
            self.assertEqual(matmul_torch_compile([[4.0]], [[2.0]]), [[8.0]])
        self.assertEqual(compiled, [torch.matmul])

    def test_torch_backend_reports_missing_optional_dependency(self):
        with patch.dict("sys.modules", {"torch": None}):
            with self.assertRaisesRegex(RuntimeError, "requires torch"):
                matmul_torch([[1.0]], [[2.0]])

    def test_numpy_backend_reports_missing_optional_dependency(self):
        with patch.dict("sys.modules", {"numpy": None}):
            with self.assertRaisesRegex(RuntimeError, "requires numpy"):
                matmul_numpy([[1.0]], [[2.0]])

    def test_causal_attention_masks_future_tokens_and_scales_scores(self):
        actual = causal_attention(
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[10.0], [20.0]],
        )
        weight = math.exp(1.0 / math.sqrt(2.0))
        self.assertEqual(actual[0], [10.0])
        self.assertAlmostEqual(actual[1][0], (10.0 + 20.0 * weight) / (1.0 + weight))
        with self.assertRaises(ValueError):
            causal_attention([[1.0]], [[1.0], [2.0]], [[1.0]])
        with self.assertRaises(ValueError):
            causal_attention([[1.0]], [[1.0, 2.0]], [[1.0]])

    def test_row_layer_norm_normalizes_each_row_and_validates_epsilon(self):
        actual = row_layer_norm([[1.0, 3.0], [5.0, 5.0]], epsilon=1e-8)
        self.assertAlmostEqual(actual[0][0], -1.0, places=7)
        self.assertAlmostEqual(actual[0][1], 1.0, places=7)
        self.assertEqual(actual[1], [0.0, 0.0])
        with self.assertRaises(ValueError):
            row_layer_norm([[1.0]], epsilon=0.0)

    def test_row_softmax_normalizes_each_row(self):
        actual = row_softmax([[0.0, 0.0], [0.0, math.log(3.0)]])
        self.assertEqual(actual[0], [0.5, 0.5])
        self.assertAlmostEqual(actual[1][0], 0.25)
        self.assertAlmostEqual(actual[1][1], 0.75)
        with self.assertRaises(ValueError):
            row_softmax([])

    def test_online_row_softmax_handles_large_values(self):
        actual = row_softmax_online([[1000.0, 1001.0], [-1000.0, -1000.0]])
        self.assertAlmostEqual(actual[0][0], 1.0 / (1.0 + math.e))
        self.assertAlmostEqual(actual[0][1], math.e / (1.0 + math.e))
        self.assertEqual(actual[1], [0.5, 0.5])

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

    def test_matmul_estimates_report_logical_work_and_reference_accesses(self):
        self.assertEqual(
            matmul_estimates(2, 3, 4),
            {
                "additions": 16,
                "multiplications": 24,
                "logical_flops": 48,
                "memory_element_accesses": {
                    "left_reads": 24,
                    "right_reads": 24,
                    "output_writes": 8,
                    "total": 56,
                },
            },
        )

    def test_csv_export_writes_one_row_per_tile(self):
        report = {
            "environment": {
                "cpu_count": 4,
                "machine": "test-machine",
                "platform": "test-platform",
                "python": "3.test",
            },
            "shape": [2, 3, 4],
            "repeats": 5,
            "tile_sensitivity": [
                {"tile": 1, "median_ms": 0.1, "max_abs_difference": 0.0},
                {"tile": 2, "median_ms": 0.2, "max_abs_difference": 1e-12},
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w+", newline="", encoding="utf-8") as output:
            write_csv(output.name, report)
            rows = list(csv.DictReader(output))
        self.assertEqual(rows[0]["m"], "2")
        self.assertEqual(rows[0]["tile"], "1")
        self.assertEqual(rows[1]["tile"], "2")
        self.assertEqual(rows[1]["max_abs_difference"], "1e-12")

    def test_regression_thresholds_validate_and_report_exceeded_tiles(self):
        with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as output:
            output.write('{"max_median_ms": 1.0, "max_abs_difference": 0.0}')
            output.flush()
            thresholds = load_thresholds(output.name)
        report = {"tile_sensitivity": [{"tile": 4, "median_ms": 1.1, "max_abs_difference": 0.0}]}
        self.assertEqual(threshold_violations(report, thresholds), ["tile 4 max_median_ms"])
        with self.assertRaises(OSError):
            load_thresholds("does-not-exist.json")

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
