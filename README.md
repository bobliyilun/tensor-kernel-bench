# tensor-kernel-bench

An experimental CPU reference lab for tensor-kernel correctness and benchmarking. The baseline compares naive and tiled matrix multiplication using deterministic inputs and no third-party dependencies.

The pure-Python implementation is deliberately a correctness oracle. Native, NumPy, and Triton backends are roadmap items rather than implied current capabilities.

## Run

```bash
python3 benchmark.py --m 24 --k 32 --n 16 --tile 8 --repeats 3
python3 -m unittest -v
```

See [ROADMAP.md](ROADMAP.md) for kernel and backend milestones.

