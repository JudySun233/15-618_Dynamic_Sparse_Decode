#include "dsd/paged_kv_cache.h"

#include <algorithm>
#include <cstddef>
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
  const int start_token = NextStartTokenForRequest(request_id);
  const auto page_offset =
      static_cast<std::size_t>(page_id) * ElementsPerPage();
  auto* summary_ptr = MutablePageSummary(page_id);

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

AppendTokenResult PagedKvCache::AppendToken(
    int request_id,
    const std::vector<float>& key,
    const std::vector<float>& value) {
  const int elements_per_token = ElementsPerToken();
  if (static_cast<int>(key.size()) != elements_per_token ||
      static_cast<int>(value.size()) != elements_per_token) {
    throw std::invalid_argument("token payload does not match expected size");
  }

  auto& request_pages = request_to_pages_[request_id];
  bool allocated_new_page = false;
  PageId page_id = -1;
  if (request_pages.empty() ||
      pages_[static_cast<std::size_t>(request_pages.back())].token_count >=
          config_.page_size) {
    page_id = page_pool_.allocate_page();
    const auto page_offset =
        static_cast<std::size_t>(page_id) * ElementsPerPage();
    pages_[static_cast<std::size_t>(page_id)] = PageDescriptor{
        page_id,
        request_id,
        NextStartTokenForRequest(request_id),
        0,
        page_offset,
        page_offset,
    };
    std::fill(MutablePageSummary(page_id), MutablePageSummary(page_id) + elements_per_token, 0.0f);
    request_pages.push_back(page_id);
    ++active_page_count_;
    allocated_new_page = true;
  } else {
    page_id = request_pages.back();
  }

  auto& page = pages_[static_cast<std::size_t>(page_id)];
  if (page.request_id != request_id || page.token_count < 0 ||
      page.token_count >= config_.page_size) {
    throw std::runtime_error("request tail page is inconsistent");
  }

  const int token_offset = page.token_count;
  auto* k_page = page_pool_.get_k_page_ptr(page_id);
  auto* v_page = page_pool_.get_v_page_ptr(page_id);
  const auto element_offset =
      static_cast<std::ptrdiff_t>(token_offset) * elements_per_token;
  std::copy(key.begin(), key.end(), k_page + element_offset);
  std::copy(value.begin(), value.end(), v_page + element_offset);

  auto* summary_ptr = MutablePageSummary(page_id);
  const float old_count = static_cast<float>(page.token_count);
  const float new_count = old_count + 1.0f;
  for (int i = 0; i < elements_per_token; ++i) {
    summary_ptr[i] =
        (summary_ptr[i] * old_count + key[static_cast<std::size_t>(i)]) /
        new_count;
  }
  ++page.token_count;

  return AppendTokenResult{page_id, token_offset, allocated_new_page};
}

std::vector<PageId> PagedKvCache::ReleaseRequest(int request_id) {
  const auto it = request_to_pages_.find(request_id);
  if (it == request_to_pages_.end()) {
    return {};
  }

  std::vector<PageId> released_pages = it->second;
  for (PageId page_id : released_pages) {
    auto& page = pages_[static_cast<std::size_t>(page_id)];
    if (page.request_id != request_id) {
      throw std::runtime_error("request page mapping is inconsistent");
    }
    std::fill(MutablePageSummary(page_id), MutablePageSummary(page_id) + ElementsPerToken(), 0.0f);
    pages_[static_cast<std::size_t>(page_id)] = MakeFreeDescriptor(page_id);
    page_pool_.free_page(page_id);
    --active_page_count_;
  }
  request_to_pages_.erase(it);
  return released_pages;
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

void PagedKvCache::CopyPageToken(
    PageId page_id,
    int token_offset,
    std::vector<float>* key,
    std::vector<float>* value) const {
  if (key == nullptr || value == nullptr) {
    throw std::invalid_argument("token copy outputs must be non-null");
  }
  const auto& page = GetPage(page_id);
  if (page.request_id == -1 || page.token_count == 0) {
    throw std::invalid_argument("cannot copy a token from a free page");
  }
  if (token_offset < 0 || token_offset >= page.token_count) {
    throw std::out_of_range("token_offset is outside the valid page prefix");
  }

  const int elements_per_token = ElementsPerToken();
  const auto element_offset =
      static_cast<std::ptrdiff_t>(token_offset) * elements_per_token;
  const auto* key_ptr = page_pool_.get_k_page_ptr(page_id) + element_offset;
  const auto* value_ptr = page_pool_.get_v_page_ptr(page_id) + element_offset;
  key->assign(key_ptr, key_ptr + elements_per_token);
  value->assign(value_ptr, value_ptr + elements_per_token);
}

std::vector<float> PagedKvCache::BuildPageSummary(PageId page_id) const {
  return CopyPageSummary(page_id);
}

std::vector<float> PagedKvCache::CopyPageSummary(PageId page_id) const {
  const auto& page = GetPage(page_id);
  const auto* summary_ptr = PageSummary(page_id);
  if (page.token_count == 0) {
    return std::vector<float>(static_cast<std::size_t>(ElementsPerToken()), 0.0f);
  }
  return std::vector<float>(summary_ptr, summary_ptr + ElementsPerToken());
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

int PagedKvCache::NextStartTokenForRequest(int request_id) const {
  const auto it = request_to_pages_.find(request_id);
  if (it == request_to_pages_.end() || it->second.empty()) {
    return 0;
  }
  const auto& last_page = GetPage(it->second.back());
  return last_page.start_token + last_page.token_count;
}

float* PagedKvCache::MutablePageSummary(PageId page_id) {
  if (page_id < 0 || page_id >= static_cast<PageId>(pages_.size())) {
    throw std::out_of_range("invalid page id");
  }
  return page_summary_pool_.data() +
         static_cast<std::ptrdiff_t>(
             static_cast<std::size_t>(page_id) * ElementsPerToken());
}

const float* PagedKvCache::PageSummary(PageId page_id) const {
  if (page_id < 0 || page_id >= static_cast<PageId>(pages_.size())) {
    throw std::out_of_range("invalid page id");
  }
  return page_summary_pool_.data() +
         static_cast<std::ptrdiff_t>(
             static_cast<std::size_t>(page_id) * ElementsPerToken());
}

}  // namespace dsd
