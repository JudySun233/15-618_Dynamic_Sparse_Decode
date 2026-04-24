#include "dsd/paged_kv_cache.h"

#include <algorithm>
#include <stdexcept>

namespace dsd {

namespace {

const std::vector<PageId>& EmptyPageList() {
  static const std::vector<PageId> empty;
  return empty;
}

}  // namespace

PagedKvCache::PagedKvCache(ModelConfig config, int capacity_pages)
    : config_(config),
      page_pool_(config, capacity_pages),
      page_summary_pool_(
          static_cast<std::size_t>(capacity_pages) *
              static_cast<std::size_t>(config.num_heads * config.head_dim),
          0.0f) {
  pages_.reserve(static_cast<std::size_t>(capacity_pages));
  for (PageId page_id = 0; page_id < capacity_pages; ++page_id) {
    pages_.push_back(MakeFreeDescriptor(page_id));
  }
}

PageId PagedKvCache::AppendPage(
    int request_id,
    const std::vector<float>& keys,
    const std::vector<float>& values,
    int token_count) {
  if (token_count <= 0 || token_count > config_.page_size) {
    throw std::invalid_argument("token_count must be in (0, page_size]");
  }

  const std::size_t expected_elements =
      static_cast<std::size_t>(token_count) * ElementsPerToken();
  if (keys.size() != expected_elements || values.size() != expected_elements) {
    throw std::invalid_argument("page payload does not match expected size");
  }

  const PageId page_id = page_pool_.allocate_page();
  const int start_token =
      static_cast<int>(request_to_pages_[request_id].size()) * config_.page_size;
  const auto page_offset =
      static_cast<std::size_t>(page_id) * ElementsPerPage();
  auto* summary_ptr = page_summary_pool_.data() +
      static_cast<std::ptrdiff_t>(static_cast<std::size_t>(page_id) * ElementsPerToken());

  std::copy(keys.begin(), keys.end(), page_pool_.get_k_page_ptr(page_id));
  std::copy(values.begin(), values.end(), page_pool_.get_v_page_ptr(page_id));
  std::fill(summary_ptr, summary_ptr + ElementsPerToken(), 0.0f);
  for (int token = 0; token < token_count; ++token) {
    const auto base = static_cast<std::size_t>(token) * ElementsPerToken();
    for (int i = 0; i < ElementsPerToken(); ++i) {
      summary_ptr[i] += keys[base + static_cast<std::size_t>(i)];
    }
  }
  const float inv_token_count = 1.0f / static_cast<float>(token_count);
  for (int i = 0; i < ElementsPerToken(); ++i) {
    summary_ptr[i] *= inv_token_count;
  }

  pages_[static_cast<std::size_t>(page_id)] = PageDescriptor{
      page_id,
      request_id,
      start_token,
      token_count,
      page_offset,
      page_offset,
  };
  request_to_pages_[request_id].push_back(page_id);
  ++active_page_count_;
  return page_id;
}

void PagedKvCache::Reset() {
  request_to_pages_.clear();
  active_page_count_ = 0;
  page_pool_.reset_pool();
  std::fill(page_summary_pool_.begin(), page_summary_pool_.end(), 0.0f);
  for (PageId page_id = 0; page_id < static_cast<PageId>(pages_.size()); ++page_id) {
    pages_[static_cast<std::size_t>(page_id)] = MakeFreeDescriptor(page_id);
  }
}

const PageDescriptor& PagedKvCache::GetPage(PageId page_id) const {
  if (page_id < 0 || page_id >= static_cast<PageId>(pages_.size())) {
    throw std::out_of_range("invalid page id");
  }
  return pages_[page_id];
}

const std::vector<PageId>& PagedKvCache::GetRequestPages(int request_id) const {
  const auto it = request_to_pages_.find(request_id);
  if (it == request_to_pages_.end()) {
    return EmptyPageList();
  }
  return it->second;
}

std::vector<float> PagedKvCache::CopyPageKeys(PageId page_id) const {
  const auto& page = GetPage(page_id);
  const auto element_count =
      static_cast<std::size_t>(page.token_count) * ElementsPerToken();
  if (element_count == 0) {
    return {};
  }

  const auto* key_ptr = page_pool_.get_k_page_ptr(page_id);
  return std::vector<float>(key_ptr, key_ptr + static_cast<std::ptrdiff_t>(element_count));
}

std::vector<float> PagedKvCache::CopyPageValues(PageId page_id) const {
  const auto& page = GetPage(page_id);
  const auto element_count =
      static_cast<std::size_t>(page.token_count) * ElementsPerToken();
  if (element_count == 0) {
    return {};
  }

  const auto* value_ptr = page_pool_.get_v_page_ptr(page_id);
  return std::vector<float>(
      value_ptr, value_ptr + static_cast<std::ptrdiff_t>(element_count));
}

std::vector<float> PagedKvCache::BuildPageSummary(PageId page_id) const {
  return CopyPageSummary(page_id);
}

std::vector<float> PagedKvCache::CopyPageSummary(PageId page_id) const {
  const auto& page = GetPage(page_id);
  const auto summary_offset =
      static_cast<std::size_t>(page_id) * ElementsPerToken();
  const auto begin =
      page_summary_pool_.begin() + static_cast<std::ptrdiff_t>(summary_offset);
  const auto end = begin + ElementsPerToken();
  if (page.token_count == 0) {
    return std::vector<float>(static_cast<std::size_t>(ElementsPerToken()), 0.0f);
  }
  return std::vector<float>(begin, end);
}

int PagedKvCache::TotalPages() const {
  return active_page_count_;
}

int PagedKvCache::ElementsPerToken() const {
  return config_.num_heads * config_.head_dim;
}

int PagedKvCache::ElementsPerPage() const {
  return config_.page_size * ElementsPerToken();
}

PageDescriptor PagedKvCache::MakeFreeDescriptor(PageId page_id) const {
  const auto page_offset =
      static_cast<std::size_t>(page_id) * ElementsPerPage();
  return PageDescriptor{
      page_id,
      -1,
      0,
      0,
      page_offset,
      page_offset,
  };
}

}  // namespace dsd
