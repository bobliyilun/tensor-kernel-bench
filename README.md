# tensor-kernel-bench

An experimental CPU reference lab for tensor-kernel correctness and benchmarking. The baseline compares naive and tiled matrix multiplication using deterministic inputs and no third-party dependencies.

The pure-Python implementation is deliberately a correctness oracle. It includes
naive and tiled kernels for conventional `[k][n]` right-hand-side storage plus
pre-transposed `[n][k]` storage, and `batched_matmul` for matching batches of
matrices. `matmul_bias(left, right, bias)` adds a length-`n` column bias while
accumulating the product. `matmul_relu(left, right)` and `matmul_gelu(left,
right)` apply ReLU and exact GELU output epilogues respectively. `row_softmax(values)`
independently normalizes each matrix row. `row_softmax_online(values)` uses an
online maximum and sum to avoid overflow on large finite logits. `row_layer_norm(values,
epsilon)` normalizes each row using its population variance. `causal_attention(query, key,
value)` applies stable scaled dot-product attention with a causal mask. Native and Triton
backends are roadmap items rather than implied current capabilities.

## Run

```bash
python3 benchmark.py --m 24 --k 32 --n 16 --tile 8 --repeats 3
python3 benchmark.py --m 256 --k 256 --n 256 --tiles 1,2,4,8,16,32 --repeats 5
python3 benchmark.py --m 256 --k 256 --n 256 --tiles 8,16,32 --repeats 5 --csv results.csv
python3 benchmark.py --m 24 --k 32 --n 16 --tiles 4,8 --repeats 3 --thresholds thresholds.json
python3 benchmark.py --m 24 --k 32 --n 16 --backend numpy
python3 benchmark.py --m 24 --k 32 --n 16 --backend torch
python3 benchmark.py --m 24 --k 32 --n 16 --backend torch-compile
python3 benchmark.py --m 24 --k 32 --n 16 --backend triton
python3 -m unittest -v
```

Use `--tiles` to measure tile-size sensitivity. Each result includes its
correctness difference from the naive reference and basic runtime context;
compare only runs with the same shape, repeat count, and environment.
It also includes exact logical arithmetic counts and element-access estimates
for the reference matmul loop. These model reads from the two inputs and one
write per output element; they are not measurements of cache or DRAM traffic.
Pass `--csv PATH` to write one flat row per measured tile, including the
shape, repeat count, timing, correctness difference, and runtime context.
Pass `--thresholds PATH` to enforce JSON ceilings such as
`{"max_median_ms": 10.0, "max_abs_difference": 1e-12}` for every requested
tile. The JSON report records the threshold verdict and exits nonzero when a
ceiling is exceeded; choose timing ceilings only for a stable, comparable
environment.

The optional NumPy backend uses `numpy.matmul` and is selected with `--backend
numpy`. Install NumPy separately (`python3 -m pip install numpy`); the default
backend remains the pure-Python reference.

The optional PyTorch eager backend uses `torch.matmul` and is selected with
`--backend torch`. Install PyTorch separately (`python3 -m pip install torch`);
it also defaults to CPU tensors and leaves the pure-Python backend unchanged.

The optional `torch.compile` comparison uses a cached compiled `torch.matmul`
and is selected with `--backend torch-compile`. It requires PyTorch 2.0 or
newer. The first correctness run triggers compilation before timed repeats, so
the reported backend timing excludes first-use compilation but includes CPU
tensor conversion.

The optional Triton backend is selected with `--backend triton`. It requires
CUDA-capable PyTorch and Triton (`python3 -m pip install torch triton`) and
uses a 16x16 float32 blocked kernel; its timing includes host-to-device tensor
conversion and the final result transfer back to CPU.

See [ROADMAP.md](ROADMAP.md) for kernel and backend milestones.
See [RESULTS.md](RESULTS.md) for the current backend comparison and its exact
hardware metadata.
