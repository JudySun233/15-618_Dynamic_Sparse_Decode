# Batched Dynamic Sparse Decode on GPUs

## Group Information
- Haojia Sun: haojias@andrew.cmu.edu
- Yuling Wang: yulingwa@andrew.cmu.edu

## Project Documents
- [Proposal](https://drive.google.com/file/d/1fHH5PrYwscMxC24i6ncw-auUZfnq9JzB/view?usp=sharing)

## What This Repo Is For
This repository is a research prototype for long-context decode on GPUs.

The main question is:

When does page-level dynamic sparsity beat dense decode once page scoring, top-k selection, KV gather, and batching overheads are included?

## Current Paths
The repo currently contains four main decode paths:

- `dense_cpu`: dense attention on CPU
- `sparse_cpu`: CPU reference sparse pipeline with `score -> topk -> gather -> attend`
- `dense_gpu`: dense CUDA baseline
- `sparse_gpu`: naive sparse CUDA path on H100/sm90, with CPU page scoring/top-k and GPU gather + sparse attention

## Build
Basic build:

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

For CUDA builds on H100, make sure `nvcc` is visible first. One working example is:

```bash
export PATH=/opt/packages/cuda/v12.4.0/bin:$PATH
export LIBRARY_PATH=/opt/packages/cuda/v12.4.0/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/packages/cuda/v12.4.0/lib64:$LD_LIBRARY_PATH

rm -rf build
cmake -S . -B build -DDSD_ENABLE_CUDA=ON -DDSD_CUDA_ARCHITECTURES=90
```

## Run
Run the demo:

```bash
./build/dsd_demo
```

Run sparse CUDA correctness test:

```bash
./build/dsd_sparse_cuda_test
```

Run dense CUDA correctness test:

```bash
./build/dsd_dense_cuda_test
```

Run the benchmark:

```bash
./build/dsd_bench [top_k_pages] [batch_size] [min_context_tokens] [max_context_tokens] [seed] [iterations] [warmup]
```

Example:

```bash
./build/dsd_bench 8 16 512 2048 7 10 2
```

Single-request example:

```bash
./build/dsd_bench 8 1 1024 1024 7 10 2
```

## Benchmark Output
`dsd_bench` now prints two tables.

`End-to-End`
- Total wall-clock time for each path
- Speedup relative to `dense_cpu`

`Stage / Kernel Breakdown`
- `dense_cpu`: CPU dense attention time
- `sparse_cpu`: CPU score, top-k, gather, attention breakdown
- `dense_gpu`: CUDA dense attention kernel time
- `sparse_gpu`: CPU score/top-k plus GPU gather/attention breakdown

It also prints an `Accuracy` block with max-absolute-difference comparisons:
- `sparse_cpu` vs `dense_cpu`
- `sparse_gpu` vs `sparse_cpu`
- `sparse_gpu` vs `dense_cpu`
- `dense_gpu` vs `dense_cpu`

## Code Layout
```text
include/dsd/
  config.h                 model/runtime configuration
  types.h                  common pipeline data structures
  paged_kv_cache.h         paged KV-cache abstraction
  reference_kernels.h      CPU reference scoring/top-k/gather/attention
  cuda_dense_attention.h   dense CUDA interface
  cuda_sparse_attention.h  sparse CUDA interface
  decode_pipeline.h        end-to-end sparse and dense decode driver
  profiler.h               lightweight stage timer
  synthetic_data.h         ragged synthetic batch generator

src/
  main.cpp
  paged_kv_cache.cpp
  reference_kernels.cpp
  decode_pipeline.cpp
  synthetic_data.cpp
  cuda/kernels.cu          dense CUDA baseline
  cuda/sparse_attention.cu sparse CUDA gather + attention

benchmarks/
  bench_decode.cpp         benchmark harness for dense/sparse CPU/GPU comparison

tests/
  test_reference.cpp
  test_dense_cuda.cpp
  test_sparse_cuda.cpp
```

## Notes
- The current `sparse_gpu` path is correctness-first and not yet batch-optimized.
- In the current implementation, sparse page scoring and top-k selection are still on CPU.
- `sparse_gpu` total time includes kernel time plus host/device copy, allocation, and synchronization overhead.
