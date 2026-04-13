#pragma once

namespace dsd {

struct ModelConfig {
  int num_heads = 4;
  int head_dim = 16;
  int page_size = 16;
  int top_k_pages = 4;
};

struct RuntimeConfig {
  int batch_size = 8;
  int seed = 7;
  bool enable_reordering = false;
  bool print_per_request = false;
};

}  // namespace dsd
