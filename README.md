# Batched Dynamic Sparse Decode on GPUs

## Group Information
- Haojia Sun: haojias@andrew.cmu.edu
- Yuling Wang: yulingwa@andrew.cmu.edu

## Project Documents
- [Proposal](https://drive.google.com/file/d/1fHH5PrYwscMxC24i6ncw-auUZfnq9JzB/view?usp=sharing)

## What This Repo Is For
This repository is structured as a standalone research prototype for long-context decode on GPUs.

The immediate goal is not to build a full serving engine. The goal is to build a clean end-to-end decode pipeline that lets us answer the systems question:

When does page-level dynamic sparsity actually beat dense decode once page scoring, top-k selection, KV gather, reordering, and batching overheads are included?

The current scaffold includes:
- A paged KV cache abstraction
- A CPU reference implementation of the full decode pipeline
- A naive sparse pipeline driver with stage-level timing
- A synthetic workload generator for ragged batched decode
- A benchmark entry point and a correctness test
- An optional CUDA target where real kernels can be added incrementally

## Build
```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run the demo:
```bash
./build/dsd_demo
```

Run the benchmark:
```bash
./build/dsd_bench [top_k_pages] [batch_size] [min_context_tokens] [max_context_tokens] [seed]
```

Example:
```bash
./build/dsd_bench 8 16 512 2048 7
```

## Code Layout
```text
include/dsd/
  config.h            model/runtime configuration
  types.h             common pipeline data structures
  paged_kv_cache.h    paged KV-cache abstraction
  reference_kernels.h CPU reference scoring/top-k/gather/attention
  decode_pipeline.h   end-to-end sparse and dense decode driver
  profiler.h          lightweight stage timer
  synthetic_data.h    ragged synthetic batch generator

src/
  main.cpp            small end-to-end demo
  paged_kv_cache.cpp
  reference_kernels.cpp
  decode_pipeline.cpp
  synthetic_data.cpp
  cuda/kernels.cu     placeholder CUDA translation unit

benchmarks/
  bench_decode.cpp    baseline timing harness

tests/
  test_reference.cpp  sparse=dense check when top-k selects all pages
```

## Recommended Development Order
### Phase 1: Correctness-first baseline
1. Keep the CPU reference path correct at all times.
2. Replace one stage at a time with CUDA while comparing against the CPU path.
3. Preserve explicit stage boundaries: `score -> select -> gather -> attend`.
4. Measure per-stage latency before trying end-to-end optimization.

### Phase 2: Naive GPU pipeline
Implement one straightforward GPU version for each stage before optimizing:
- `page_scoring`
- `topk_selection`
- `kv_gather`
- `sparse_attention`

The first GPU version should favor debuggability over peak performance.

### Phase 3: Optimization work
After the naive path runs end-to-end, optimize in this order:
- KV gather layout and coalescing
- Reordering selected pages to improve locality
- Better thread/block mapping for ragged requests
- Batched scheduling across requests
- Selection reuse or page grouping across nearby decode steps

## Suggested Interfaces To Preserve
Keep these boundaries stable even when the internals change:
- `PagedKvCache`: owns physical page storage
- `RequestState`: logical request metadata and candidate pages
- `ScorePagesCpu / ScorePagesCuda`
- `SelectTopKPagesCpu / SelectTopKPagesCuda`
- `GatherPagesCpu / GatherPagesCuda`
- `SparseAttentionCpu / SparseAttentionCuda`
- `DecodePipeline`: orchestrates the stages and records latency

If you preserve those interfaces, you can swap implementations without rewriting the whole experiment harness.

## Week 1 Milestone
The best first milestone is:

1. Make the current CPU reference compile and pass tests.
2. Implement a dense GPU attention baseline with the same data layout.
3. Implement a naive page scoring kernel.
4. Add timing and correctness checks for each stage.

That will give you a stable backbone for the rest of the project.
