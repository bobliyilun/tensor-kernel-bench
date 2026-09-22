# Backend comparison

This is a reproducible snapshot, not a cross-machine ranking.  Timings are
median wall-clock milliseconds over five repeats of a `128 x 128` by `128 x
128` float matrix multiplication with deterministic seed `0`.  Optional
backend timings include the conversion/transfer behavior documented in the
README.

## 2026-09-22 (Asia/Shanghai)

Hardware and runtime: macOS 26.5.2, `arm64`, 8 logical CPUs, Python 3.9.6.
No CUDA device was available, and NumPy, PyTorch, and Triton were not installed.

| Backend | Median ms | Maximum absolute difference | Status |
| --- | ---: | ---: | --- |
| Python reference | 165.177 | 0.0 | measured |
| Tiled Python (tile 8) | 271.257 | 0.0 | measured |
| Tiled Python (tile 16) | 205.884 | 0.0 | measured |
| Tiled Python (tile 32) | 185.907 | 0.0 | measured |
| NumPy | — | — | not measured: NumPy unavailable |
| PyTorch eager | — | — | not measured: PyTorch unavailable |
| `torch.compile` | — | — | not measured: PyTorch unavailable |
| Triton | — | — | not measured: PyTorch and Triton unavailable; CUDA required |

Run the same command on a machine with optional dependencies installed to add
a comparable row: `python3 benchmark.py --m 128 --k 128 --n 128 --tile 16
--repeats 5 --seed 0 --backend BACKEND`.
