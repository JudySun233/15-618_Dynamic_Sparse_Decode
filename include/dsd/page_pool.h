#pragma once

#include <cstdint>
#include <vector>

#include "dsd/config.h"
#include "dsd/types.h"

namespace dsd {

class PagePool {
 public:
  PagePool(ModelConfig config, int capacity_pages, bool allocate_storage = true);

  PageId allocate_page();
  void free_page(PageId page_id);

  float* get_k_page_ptr(PageId page_id);
  const float* get_k_page_ptr(PageId page_id) const;

  float* get_v_page_ptr(PageId page_id);
  const float* get_v_page_ptr(PageId page_id) const;

  void reset_pool();

  int capacity_pages() const { return capacity_pages_; }
  bool is_allocated(PageId page_id) const;
  int elements_per_token() const;
  int elements_per_page() const;

  const std::vector<float>& key_storage() const { return key_storage_; }
  const std::vector<float>& value_storage() const { return value_storage_; }
  bool has_storage() const { return has_storage_; }

 private:
  void ValidatePageId(PageId page_id) const;
  void ValidateAllocated(PageId page_id) const;
  void ValidateStorage() const;
  void InitializeFreeList();

  ModelConfig config_;
  int capacity_pages_ = 0;
  std::vector<PageId> free_list_;
  std::vector<std::uint8_t> is_allocated_;
  std::vector<float> key_storage_;
  std::vector<float> value_storage_;
  bool has_storage_ = true;
};

}  // namespace dsd
