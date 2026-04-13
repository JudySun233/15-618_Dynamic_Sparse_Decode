#pragma once

#include <vector>

namespace dsd {

using PageId = int;

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

struct DenseBatchResult {
  std::vector<AttentionResult> outputs;
  double kernel_ms = 0.0;
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
};

struct BatchDecodeResult {
  std::vector<SparseDecodeResult> per_request;
  StageTimings aggregate_timings;
};

}  // namespace dsd
