#pragma once

#include <vector>

namespace dsd {

using PageId = int;

struct AppendTokenResult {
  PageId page_id = -1;
  int token_offset = 0;
  bool allocated_new_page = false;
};

struct RequestState {
  int request_id = -1;
  std::vector<float> query;
  std::vector<PageId> candidate_page_ids;
  int context_tokens = 0;
};

struct PageScore {
  PageId page_id = -1;
  float score = 0.0f;
};

struct GatheredPages {
  std::vector<PageId> page_ids;
  std::vector<int> token_offsets;
  std::vector<int> token_counts;
  std::vector<float> keys;
  std::vector<float> values;
};

struct AttentionResult {
  std::vector<float> output;
};

struct RuntimeOverheadTimings {
  double time_malloc_ms = 0.0;
  double time_memcpy_h2d_ms = 0.0;
  double time_memcpy_d2h_ms = 0.0;
  double time_free_ms = 0.0;
  double time_kernel_launch_ms = 0.0;
  double time_sync_ms = 0.0;
  double time_prepare_sparse_layout_ms = 0.0;
};

struct DenseBatchResult {
  std::vector<AttentionResult> outputs;
  double kernel_ms = 0.0;
  RuntimeOverheadTimings runtime_overheads;
};

struct StageTimings {
  double page_scoring_ms = 0.0;
  double topk_ms = 0.0;
  double gather_ms = 0.0;
  double attention_ms = 0.0;
  double total_ms = 0.0;
};

struct SparseDecodeResult {
  int request_id = -1;
  std::vector<PageScore> scores;
  std::vector<PageId> selected_page_ids;
  AttentionResult output;
  StageTimings timings;
  RuntimeOverheadTimings runtime_overheads;
};

struct BatchDecodeResult {
  std::vector<SparseDecodeResult> per_request;
  StageTimings aggregate_timings;
};

struct SparseBatchCudaResult {
  std::vector<SparseDecodeResult> per_request;
  StageTimings aggregate_timings;
  RuntimeOverheadTimings runtime_overheads;
  double kernel_ms = 0.0;
};

}  // namespace dsd
