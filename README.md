# Batched Dynamic Sparse Decode on GPUs

## Group Information
- Haojia Sun: haojias@andrew.cmu.edu
- Yuling Wang: yulingwa@andrew.cmu.edu

## Project Documents
- [Proposal](https://drive.google.com/file/d/1fHH5PrYwscMxC24i6ncw-auUZfnq9JzB/view?usp=sharing)
- [Milestone Report](https://drive.google.com/file/d/1VijTbikrY2i88_Sm0u1dNoGkF70GW7i0/view?usp=sharing)

## What This Repo Is For
This repository is a research prototype for long-context decode and
continuous batching on GPUs.

The main question is:

When does page-level dynamic sparsity beat dense decode once page scoring, top-k selection, KV gather, and batching overheads are included?

## Current Paths
The repo currently contains four main decode paths:

- `dense_cpu`: dense attention on CPU
- `sparse_cpu`: CPU reference sparse pipeline with `score -> topk -> gather -> attend`
- `dense_gpu`: dense CUDA baseline
- `sparse_gpu`: sparse CUDA path on H100/sm90, with GPU page scoring, GPU top-k selection, GPU gather, and GPU sparse attention; sparse layout preparation still includes host-side metadata work

It also contains continuous-serving benchmark paths:

- `continuous_sparse`: continuous batching over the sparse GPU decode path
- `continuous_dense_gpu`: continuous batching over the dense CUDA decode path
- `serial_sparse`: one-active-request sparse baseline used for continuous-vs-serial comparison

## Memory / Cache Infrastructure
The sparse decode pipeline is built on top of explicit page-pool abstractions for both host and device memory.

- `PagePool`: contiguous host-side K/V storage with page allocation, free-list reuse, and slot-based addressing
- `PagedKvCache`: logical request-to-page mapping on top of `PagePool`, including page descriptors, page summaries, and request page order tracking
- `DevicePagePool`: CUDA-resident mirror for page payloads and metadata used by the sparse GPU path

These modules make it possible to separate logical page selection from the physical storage layout used by CPU and GPU decode paths.

For continuous batching, request prompts are admitted into the paged cache, active
requests are packed into per-step batches, generated tokens are appended back to
the cache, and finished requests release their pages for reuse. The default
prompt admission mode is direct GPU upload.

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
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDSD_ENABLE_CUDA=ON -DDSD_CUDA_ARCHITECTURES=90
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
./build/dsd_bench [top_k_pages] [batch_size] [min_context_tokens] [max_context_tokens] [seed] [iterations] [warmup] [num_heads] [head_dim] [page_size]
```

Example:

```bash
./build/dsd_bench 8 16 512 2048 7 10 2 32 128 16
```

Single-request example:

```bash
./build/dsd_bench 8 1 1024 1024 7 10 2
```

Run the continuous batching benchmark:

```bash
./build/dsd_continuous_bench \
  [num_requests] [max_active_requests] [arrival_window] \
  [min_prompt_tokens] [max_prompt_tokens] \
  [min_decode_steps] [max_decode_steps] [seed] \
  [num_heads] [head_dim] [page_size] [top_k_pages] \
  [admission_mode] [preadmit_prompts] [precompute_decode_payloads] \
  [gpu_synthetic_decode_append] [lazy_release] \
  [run_batch_output_mode] [run_batch_timing_mode] [benchmark_mode]
```

Important continuous benchmark modes:

- `admission_mode`: `0=cpu_cache`, `1=direct_gpu_upload`, `2=synthetic_gpu_prefill`
- `run_batch_output_mode`: `0=full_outputs`, `1=outputs_only`, `2=no_outputs`
- `run_batch_timing_mode`: `0=host_wall`, `1=kernel_events`, `2=disabled`
- `benchmark_mode`: `0=sparse`, `1=dense_gpu`

Example sparse continuous run:

```bash
./build/dsd_continuous_bench 128 32 64 1024 4096 16 64 7 32 128 16 8
```

Example dense GPU continuous run:

```bash
./build/dsd_continuous_bench 128 32 64 1024 4096 16 64 7 32 128 16 8 1 0 0 0 0 1 1 1
```

## Benchmark Output
`dsd_bench` now prints three sections before the accuracy summary.

`Analytical Ceiling`
- exact batch totals for context tokens, candidate pages, and selected tokens
- dense, current sparse, and target sparse byte/FLOP estimates
- H100 bandwidth and FP32 lower-bound times

`End-to-End`
- Total wall-clock time for each path
- Speedup relative to `dense_cpu`

`Stage / Kernel Breakdown`
- `dense_cpu`: CPU dense attention time
- `sparse_cpu`: CPU score, top-k, gather, attention breakdown
- `dense_gpu`: CUDA dense attention kernel time
- `sparse_gpu`: batched GPU score/top-k plus fused sparse attention breakdown

It also prints an `Accuracy` block with max-absolute-difference comparisons:
- `sparse_cpu` vs `dense_cpu`
- `sparse_gpu` vs `sparse_cpu`
- `sparse_gpu` vs `dense_cpu`
- `dense_gpu` vs `dense_cpu`

`dsd_continuous_bench` prints key-value output for continuous serving runs.
The sparse mode reports both `continuous_sparse_*` and `serial_sparse_*`
statistics plus `continuous_vs_serial_speedup`. Dense GPU mode reports
`continuous_dense_gpu_*` statistics.

Important continuous fields include:

- `*_tokens_per_second`, `*_avg_active_batch_size`, `*_avg_step_ms`, `*_p95_step_ms`
- `*_measured_end_to_end_ms`, `*_total_ms`, `*_run_batch_wall_ms`, `*_outside_run_batch_ms`
- `*_admission_ms`, `*_prompt_preadmit_ms`, `*_runtime_admission_ms`
- `*_append_sync_ms`, `*_release_sync_ms`, `*_serving_loop_other_ms`
- `*_avg_score_ms`, `*_avg_topk_ms`, `*_avg_attention_ms`, `*_avg_kernel_ms`
- `*_avg_h2d_ms`, `*_avg_d2h_ms`, `*_avg_launch_ms`, `*_avg_sync_ms`
- `*_device_h2d_bytes`, `*_device_d2h_bytes`, and transfer call counters

## Report Experiments
The `experiment/` directory contains report-oriented sweep scripts and matching
plot scripts. Each shell script writes raw stdout and CSV files under
`results/report_experiments/`; each plotting script reads the CSV and writes
figures under that experiment's `figures/` directory.

```bash
./experiment/00_backend_total_time.sh
./experiment/01_decode_context_length_sweep.sh
./experiment/02_decode_batch_scaling_sweep.sh
./experiment/03_decode_topk_sparsity_sweep.sh
./experiment/04_decode_page_size_sweep.sh
./experiment/05_continuous_batching_load_sweep.sh
./experiment/06_continuous_runtime_options_ablation.sh
./experiment/07_attention_backend_breakdown.sh
```

Plotting example:

```bash
./experiment/plot_05_continuous_batching_load_sweep.py
```

See `experiment/README.md` for the report question and key CSV columns for each
sweep.

## Code Layout
```text
include/dsd/
  config.h                 model/runtime configuration
  cuda_utils.h             CUDA helpers and checked device buffers
  types.h                  common pipeline data structures
  page_pool.h              host-side page allocator for contiguous KV storage
  paged_kv_cache.h         paged KV-cache abstraction
  device_page_pool.h       CUDA-resident page pool and page metadata storage
  reference_kernels.h      CPU reference scoring/top-k/gather/attention
  cuda_dense_attention.h   dense CUDA interface
  cuda_sparse_attention.h  sparse CUDA interface
  decode_pipeline.h        end-to-end sparse and dense decode driver
  continuous_batching.h    synthetic continuous-serving workload and runners
  profiler.h               lightweight stage timer
  synthetic_data.h         ragged synthetic batch generator

src/
  main.cpp
  page_pool.cpp
  paged_kv_cache.cpp
  reference_kernels.cpp
  decode_pipeline.cpp
  continuous_batching.cpp
  synthetic_data.cpp
  device_page_pool.cpp     CUDA-backed page-pool implementation
  cuda_dense_attention_stub.cpp
  cuda_sparse_attention_stub.cpp
  cuda/kernels.cu               dense CUDA baseline
  cuda/sparse_batched_attention.cu batched sparse CUDA score/top-k/fused-attention

benchmarks/
  bench_decode.cpp         benchmark harness for dense/sparse CPU/GPU comparison
  bench_continuous_decode.cpp continuous batching benchmark harness

experiment/
  README.md                report experiment guide
  *.sh                     reproducible experiment sweeps
  plot_*.py                CSV-to-figure plotting scripts

tests/
  test_page_pool.cpp
  test_reference.cpp
  test_device_page_pool.cpp
  test_dense_cuda.cpp
  test_sparse_cuda.cpp
  test_continuous_batching.cpp
```

## Notes
- `dense_gpu` and `sparse_gpu` both use persistent HBM contexts for static KV/page data.
- `sparse_gpu` is now batch-oriented; gather is fused into attention in the fast path, so `gather_ms` is expected to be zero.
- `sparse_gpu` total time still includes host/device copy, kernel launch, synchronization, and sparse layout preparation overhead.
- Continuous sparse and dense GPU benchmarks share the same synthetic workload generator and request admission/release model.
- Continuous sparse reports both continuous batching and one-active-request serial sparse results; dense GPU mode reports only the dense continuous path.
