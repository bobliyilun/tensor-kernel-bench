"""Benchmark the matrix-multiplication reference kernels."""

import argparse
import json
import random
import statistics
import time

from kernels import matmul, max_abs_difference, tiled_matmul


def random_matrix(rows: int, columns: int, rng: random.Random) -> list:
    return [[rng.uniform(-1.0, 1.0) for _ in range(columns)] for _ in range(rows)]


def median_ms(function, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=24)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--tile", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.m, args.k, args.n, args.tile, args.repeats) <= 0:
        parser.error("dimensions, tile, and repeats must be positive")

    rng = random.Random(args.seed)
    left = random_matrix(args.m, args.k, rng)
    right = random_matrix(args.k, args.n, rng)
    reference = matmul(left, right)
    tiled = tiled_matmul(left, right, args.tile)
    report = {
        "shape": [args.m, args.k, args.n],
        "tile": args.tile,
        "repeats": args.repeats,
        "max_abs_difference": max_abs_difference(reference, tiled),
        "naive_median_ms": median_ms(lambda: matmul(left, right), args.repeats),
        "tiled_median_ms": median_ms(lambda: tiled_matmul(left, right, args.tile), args.repeats),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

