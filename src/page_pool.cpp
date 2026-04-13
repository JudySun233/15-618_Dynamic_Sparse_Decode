#include "dsd/page_pool.h"

#include <stdexcept>

namespace dsd {

namespace {

int ValidateCapacityPages(int capacity_pages) {
  if (capacity_pages < 0) {
    throw std::invalid_argument("capacity_pages must be non-negative");
  }
  return capacity_pages;
}

}  // namespace

PagePool::PagePool(ModelConfig config, int capacity_pages)
    : config_(config),
      capacity_pages_(ValidateCapacityPages(capacity_pages)),
      is_allocated_(static_cast<std::size_t>(capacity_pages_), 0),
      key_storage_(
          static_cast<std::size_t>(capacity_pages_) * elements_per_page(),
          0.0f),
      value_storage_(
          static_cast<std::size_t>(capacity_pages_) * elements_per_page(),
          0.0f) {
  InitializeFreeList();
}

PageId PagePool::allocate_page() {
  if (free_list_.empty()) {
    throw std::runtime_error("page pool is exhausted");
  }

  const PageId page_id = free_list_.back();
  free_list_.pop_back();
  is_allocated_[static_cast<std::size_t>(page_id)] = 1;
  return page_id;
}

void PagePool::free_page(PageId page_id) {
  ValidatePageId(page_id);
  ValidateAllocated(page_id);
  is_allocated_[static_cast<std::size_t>(page_id)] = 0;
  free_list_.push_back(page_id);
}

float* PagePool::get_k_page_ptr(PageId page_id) {
  ValidatePageId(page_id);
  ValidateAllocated(page_id);
  return key_storage_.data() +
         static_cast<std::ptrdiff_t>(page_id) * elements_per_page();
}

const float* PagePool::get_k_page_ptr(PageId page_id) const {
  ValidatePageId(page_id);
  ValidateAllocated(page_id);
  return key_storage_.data() +
         static_cast<std::ptrdiff_t>(page_id) * elements_per_page();
}

float* PagePool::get_v_page_ptr(PageId page_id) {
  ValidatePageId(page_id);
  ValidateAllocated(page_id);
  return value_storage_.data() +
         static_cast<std::ptrdiff_t>(page_id) * elements_per_page();
}

const float* PagePool::get_v_page_ptr(PageId page_id) const {
  ValidatePageId(page_id);
  ValidateAllocated(page_id);
  return value_storage_.data() +
         static_cast<std::ptrdiff_t>(page_id) * elements_per_page();
}

void PagePool::reset_pool() {
  std::fill(is_allocated_.begin(), is_allocated_.end(), 0);
  InitializeFreeList();
}

bool PagePool::is_allocated(PageId page_id) const {
  ValidatePageId(page_id);
  return is_allocated_[static_cast<std::size_t>(page_id)] != 0;
}

int PagePool::elements_per_token() const {
  return config_.num_heads * config_.head_dim;
}

int PagePool::elements_per_page() const {
  return config_.page_size * elements_per_token();
}

void PagePool::ValidatePageId(PageId page_id) const {
  if (page_id < 0 || page_id >= capacity_pages_) {
    throw std::out_of_range("invalid page id");
  }
}

void PagePool::ValidateAllocated(PageId page_id) const {
  if (!is_allocated_[static_cast<std::size_t>(page_id)]) {
    throw std::runtime_error("page is not currently allocated");
  }
}

void PagePool::InitializeFreeList() {
  free_list_.clear();
  free_list_.reserve(static_cast<std::size_t>(capacity_pages_));
  for (PageId page_id = 0; page_id < capacity_pages_; ++page_id) {
    free_list_.push_back(page_id);
  }
}

}  // namespace dsd
