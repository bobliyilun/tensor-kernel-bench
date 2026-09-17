"""Benchmark the matrix-multiplication reference kernels."""

import argparse
import csv
import json
import os
import platform
import random
import statistics
import time

from kernels import matmul, matmul_numpy, matmul_torch, max_abs_difference, tiled_matmul


def random_matrix(rows: int, columns: int, rng: random.Random) -> list:
    return [[rng.uniform(-1.0, 1.0) for _ in range(columns)] for _ in range(rows)]


def median_ms(function, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def tile_sizes(value: str) -> list[int]:
    try:
        sizes = [int(size) for size in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("tiles must be comma-separated integers") from error
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("tiles must be positive")
    return sizes


def matmul_estimates(m: int, k: int, n: int) -> dict:
    """Return logical arithmetic and element-access estimates for ``m x k`` by ``k x n``.

    Memory values model the reference loop: each multiply reads one left and one
    right element, and each output element is written once.  They are element
    counts, not measured hardware traffic or byte counts.
    """
    products = m * k * n
    return {
        "additions": m * n * (k - 1),
        "multiplications": products,
        "logical_flops": products * 2,
        "memory_element_accesses": {
            "left_reads": products,
            "right_reads": products,
            "output_writes": m * n,
            "total": products * 2 + m * n,
        },
    }


def write_csv(path: str, report: dict) -> None:
    """Write one flat CSV row for every measured tile size."""
    fields = [
        "m",
        "k",
        "n",
        "tile",
        "repeats",
        "median_ms",
        "max_abs_difference",
        "python",
        "platform",
        "machine",
        "cpu_count",
    ]
    environment = report["environment"]
    m, k, n = report["shape"]
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for measurement in report["tile_sensitivity"]:
            writer.writerow(
                {
                    "m": m,
                    "k": k,
                    "n": n,
                    "tile": measurement["tile"],
                    "repeats": report["repeats"],
                    "median_ms": measurement["median_ms"],
                    "max_abs_difference": measurement["max_abs_difference"],
                    **environment,
                }
            )


def load_thresholds(path: str) -> dict:
    """Read numeric benchmark ceilings from a JSON file."""
    with open(path, encoding="utf-8") as source:
        thresholds = json.load(source)
    allowed = {"max_median_ms", "max_abs_difference"}
    if not isinstance(thresholds, dict) or not thresholds or set(thresholds) - allowed:
        raise ValueError("thresholds must contain max_median_ms and/or max_abs_difference")
    if any(not isinstance(value, (int, float)) or value < 0 for value in thresholds.values()):
        raise ValueError("thresholds must be non-negative numbers")
    return thresholds


def threshold_violations(report: dict, thresholds: dict) -> list[str]:
    """Return measured tile values that exceed configured ceilings."""
    violations = []
    for measurement in report["tile_sensitivity"]:
        for key in thresholds:
            metric = "median_ms" if key == "max_median_ms" else key
            if measurement[metric] > thresholds[key]:
                violations.append(f"tile {measurement['tile']} {key}")
    return violations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=24)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--tile", type=int, default=8)
    parser.add_argument(
        "--tiles", type=tile_sizes, help="comma-separated tile sizes to compare, such as 1,2,4,8"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", help="write tile measurements to this CSV file")
    parser.add_argument("--thresholds", help="JSON file with benchmark ceilings")
    parser.add_argument("--backend", choices=("python", "numpy", "torch"), default="python")
    args = parser.parse_args()
    if min(args.m, args.k, args.n, args.tile, args.repeats) <= 0:
        parser.error("dimensions, tile, and repeats must be positive")

    rng = random.Random(args.seed)
    left = random_matrix(args.m, args.k, rng)
    right = random_matrix(args.k, args.n, rng)
    reference = matmul(left, right)
    tiled = tiled_matmul(left, right, args.tile)
    backend = {"python": matmul, "numpy": matmul_numpy, "torch": matmul_torch}[args.backend]
    backend_result = backend(left, right)
    tiles = args.tiles or [args.tile]
    report = {
        "environment": {
            "cpu_count": os.cpu_count(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "shape": [args.m, args.k, args.n],
        "backend": args.backend,
        "estimates": matmul_estimates(args.m, args.k, args.n),
        "tile": args.tile,
        "tile_sensitivity": [
            {
                "max_abs_difference": max_abs_difference(reference, tiled_matmul(left, right, tile)),
                "median_ms": median_ms(lambda: tiled_matmul(left, right, tile), args.repeats),
                "tile": tile,
            }
            for tile in tiles
        ],
        "repeats": args.repeats,
        "max_abs_difference": max_abs_difference(reference, tiled),
        "backend_max_abs_difference": max_abs_difference(reference, backend_result),
        "backend_median_ms": median_ms(lambda: backend(left, right), args.repeats),
        "naive_median_ms": median_ms(lambda: matmul(left, right), args.repeats),
        "tiled_median_ms": median_ms(lambda: tiled_matmul(left, right, args.tile), args.repeats),
    }
    if args.thresholds:
        try:
            thresholds = load_thresholds(args.thresholds)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        violations = threshold_violations(report, thresholds)
        report["thresholds"] = {"path": args.thresholds, **thresholds, "passed": not violations, "violations": violations}
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.csv:
        write_csv(args.csv, report)
    if args.thresholds and violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
