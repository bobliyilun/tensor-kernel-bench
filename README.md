# tensor-kernel-bench

An experimental CPU reference lab for tensor-kernel correctness and benchmarking. The baseline compares naive and tiled matrix multiplication using deterministic inputs and no third-party dependencies.

The pure-Python implementation is deliberately a correctness oracle. It includes
naive and tiled kernels for conventional `[k][n]` right-hand-side storage plus
pre-transposed `[n][k]` storage, and `batched_matmul` for matching batches of
matrices. `matmul_bias(left, right, bias)` adds a length-`n` column bias while
accumulating the product. `matmul_relu(left, right)` and `matmul_gelu(left,
right)` apply ReLU and exact GELU output epilogues respectively. Native, NumPy,
and Triton backends are roadmap items rather than implied current capabilities.

## Run

```bash
python3 benchmark.py --m 24 --k 32 --n 16 --tile 8 --repeats 3
python3 benchmark.py --m 256 --k 256 --n 256 --tiles 1,2,4,8,16,32 --repeats 5
python3 -m unittest -v
```

Use `--tiles` to measure tile-size sensitivity. Each result includes its
correctness difference from the naive reference and basic runtime context;
compare only runs with the same shape, repeat count, and environment.

See [ROADMAP.md](ROADMAP.md) for kernel and backend milestones.
