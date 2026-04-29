# Report experiment scripts

These scripts sweep only arguments already exposed by the benchmark binaries.
Each script writes raw stdout plus a CSV under `results/report_experiments/`.

## Experiments worth including

0. `00_backend_total_time.sh`
   - Report question: what is the direct end-to-end total-time comparison across dense/sparse and CPU/GPU decode paths?
   - Key columns: `path`, `total_ms`, `speedup_vs_dense_cpu`.

1. `01_decode_context_length_sweep.sh`
   - Report question: at what context length does sparse decode start to pay for its page scoring/top-k overhead?
   - Key columns: `dense_gpu_total_ms`, `sparse_gpu_total_ms`, `sparse_target_over_dense_bytes`, `sparse_gpu_kernel_ms`.

2. `02_decode_batch_scaling_sweep.sh`
   - Report question: how much does batching amortize launch/layout overhead and improve GPU occupancy?
   - Key columns: `batch_size`, `sparse_gpu_total_ms`, `sparse_gpu_total_overhead_ms`, `sparse_gpu_kernel_ms`.

3. `03_decode_topk_sparsity_sweep.sh`
   - Report question: how does the sparsity knob trade selected tokens, runtime, and sparse-vs-dense numerical difference?
   - Key columns: `top_k_pages`, `total_selected_tokens`, `sparse_target_over_dense_bytes`, `avg_max_abs_diff_sparse_gpu_vs_dense_cpu`.

4. `04_decode_page_size_sweep.sh`
   - Report question: how sensitive is the implementation to page granularity?
   - Key columns: `page_size`, `dense_gpu_total_ms`, `sparse_gpu_total_ms`, `total_candidate_pages`, `total_selected_tokens`, `sparse_gpu_score_ms`, `sparse_gpu_topk_ms`, `sparse_gpu_attention_ms`.

5. `05_continuous_batching_load_sweep.sh`
   - Report question: how do request concurrency, context length, and arrival burstiness affect continuous batching throughput?
   - Key columns: `min_prompt_tokens`, `max_prompt_tokens`, `max_active_requests`, `arrival_window`, `continuous_tokens_per_second`, `continuous_vs_serial_speedup`, `avg_active_batch_size`, `avg_step_ms` (per-step total), `avg_kernel_ms`, `p95_step_ms`, `total_ms`.
   - The concurrency sweep fixes `min_prompt_tokens=max_prompt_tokens` for each value in `CONTEXT_LENGTHS` (default: `1024 2048 4096 8192`) so the plot can show speedup vs concurrency by context length.

6. `06_continuous_runtime_options_ablation.sh`
   - Report question: which continuous-serving runtime options move overhead, and where is that overhead concentrated under different workload conditions?
   - Key columns: `condition`, `variant`, `benchmark_mode`, `tokens_per_second`, `avg_h2d_ms`, `avg_d2h_ms`, `avg_launch_ms`, `avg_sync_ms`, `avg_prepare_sparse_layout_ms`, `total_runtime_overhead_ms`, `device_h2d_bytes`, `device_d2h_bytes`.

7. `07_attention_backend_breakdown.sh`
   - Report question: how do dense/sparse and CPU/GPU attention paths split time across score, top-k, gather, and attend stages?
   - Key columns: `path`, `score_ms`, `topk_ms`, `gather_ms`, `attention_ms`, `gpu_kernel_ms`, `gpu_kernel_other_ms`.

## Running

```bash
cd /jet/home/ywang94/project/15-618_Dynamic_Sparse_Decode
./experiment/00_backend_total_time.sh
./experiment/01_decode_context_length_sweep.sh
./experiment/02_decode_batch_scaling_sweep.sh
./experiment/03_decode_topk_sparsity_sweep.sh
./experiment/04_decode_page_size_sweep.sh
./experiment/05_continuous_batching_load_sweep.sh
./experiment/06_continuous_runtime_options_ablation.sh
./experiment/07_attention_backend_breakdown.sh
```

Useful overrides:

```bash
ITERATIONS=3 WARMUP=1 RESULTS_DIR=results/report_experiments_quick \
  ./experiment/03_decode_topk_sparsity_sweep.sh
```

## Plotting

Each experiment has a matching matplotlib plotting script. Run the experiment
first, then run its plot script from the repo root:

```bash
./experiment/plot_00_backend_total_time.py
./experiment/plot_01_decode_context_length_sweep.py
./experiment/plot_02_decode_batch_scaling_sweep.py
./experiment/plot_03_decode_topk_sparsity_sweep.py
./experiment/plot_04_decode_page_size_sweep.py
./experiment/plot_05_continuous_batching_load_sweep.py
./experiment/plot_06_continuous_runtime_options_ablation.py
./experiment/plot_07_attention_backend_breakdown.py
```

By default each script reads the CSV under `results/report_experiments/...`
and writes PNG figures to that experiment's `figures/` directory. If you
used a custom `RESULTS_DIR`, pass the CSV explicitly:

```bash
./experiment/plot_03_decode_topk_sparsity_sweep.py \
  --csv results/report_experiments_quick/03_decode_topk_sparsity_sweep/decode_topk_sparsity_sweep.csv \
  --out-dir results/report_experiments_quick/03_decode_topk_sparsity_sweep/figures
```

Use `--formats png,svg` if you need additional output formats.
