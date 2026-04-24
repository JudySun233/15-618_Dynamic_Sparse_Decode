#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

#include "dsd/config.h"
#include "dsd/page_pool.h"
#include "dsd/types.h"

namespace dsd {

struct PageDescriptor {
  PageId id = -1;
  int request_id = -1;
  int start_token = 0;
  int token_count = 0;
  std::size_t k_offset = 0;
  std::size_t v_offset = 0;
};

class PagedKvCache {
 public:
  explicit PagedKvCache(ModelConfig config, int capacity_pages = 1024);

  PageId AppendPage(
      int request_id,
      const std::vector<float>& keys,
      const std::vector<float>& values,
      int token_count);

  AppendTokenResult AppendToken(
      int request_id,
      const std::vector<float>& key,
      const std::vector<float>& value);

  std::vector<PageId> ReleaseRequest(int request_id);

  void Reset();

  const PageDescriptor& GetPage(PageId page_id) const;
  const std::vector<PageId>& GetRequestPages(int request_id) const;
  const std::vector<PageDescriptor>& Pages() const { return pages_; }
  const std::vector<float>& KeyPool() const { return page_pool_.key_storage(); }
  const std::vector<float>& ValuePool() const { return page_pool_.value_storage(); }
  const std::vector<float>& PageSummaryPool() const { return page_summary_pool_; }

  std::vector<float> CopyPageKeys(PageId page_id) const;
  std::vector<float> CopyPageValues(PageId page_id) const;
  void CopyPageToken(
      PageId page_id,
      int token_offset,
      std::vector<float>* key,
      std::vector<float>* value) const;
  std::vector<float> CopyPageSummary(PageId page_id) const;
  std::vector<float> BuildPageSummary(PageId page_id) const;

  int TotalPages() const;
  int CapacityPages() const { return page_pool_.capacity_pages(); }
  int ElementsPerToken() const;
  int ElementsPerPage() const;

 private:
  PageDescriptor MakeFreeDescriptor(PageId page_id) const;
  int NextStartTokenForRequest(int request_id) const;
  float* MutablePageSummary(PageId page_id);
  const float* PageSummary(PageId page_id) const;

  ModelConfig config_;
  PagePool page_pool_;
  std::vector<PageDescriptor> pages_;
  std::vector<float> page_summary_pool_;
  std::unordered_map<int, std::vector<PageId>> request_to_pages_;
  int active_page_count_ = 0;
};

}  // namespace dsd
