#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "dsd/config.h"
#include "dsd/page_pool.h"

namespace {

template <typename Fn>
bool ExpectThrows(Fn&& fn) {
  try {
    fn();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

}  // namespace

int main() {
  dsd::ModelConfig config;
  config.num_heads = 2;
  config.head_dim = 4;
  config.page_size = 3;

  dsd::PagePool pool(config, 3);
  if (pool.capacity_pages() != 3) {
    std::cerr << "unexpected pool capacity\n";
    return 1;
  }

  if (pool.elements_per_token() != 8 || pool.elements_per_page() != 24) {
    std::cerr << "unexpected element counts\n";
    return 1;
  }

  const auto first = pool.allocate_page();
  const auto second = pool.allocate_page();
  const auto third = pool.allocate_page();
  if (first != 2 || second != 1 || third != 0) {
    std::cerr << "page allocation did not follow LIFO order\n";
    return 1;
  }

  if (!pool.is_allocated(first) || !pool.is_allocated(second) ||
      !pool.is_allocated(third)) {
    std::cerr << "allocated pages were not marked live\n";
    return 1;
  }

  const auto* first_k = pool.get_k_page_ptr(first);
  const auto* first_v = pool.get_v_page_ptr(first);
  const auto* second_k = pool.get_k_page_ptr(second);
  const auto* second_v = pool.get_v_page_ptr(second);
  if ((first_k - pool.key_storage().data()) !=
          static_cast<std::ptrdiff_t>(first * pool.elements_per_page()) ||
      (first_v - pool.value_storage().data()) !=
          static_cast<std::ptrdiff_t>(first * pool.elements_per_page()) ||
      (second_k - pool.key_storage().data()) !=
          static_cast<std::ptrdiff_t>(second * pool.elements_per_page()) ||
      (second_v - pool.value_storage().data()) !=
          static_cast<std::ptrdiff_t>(second * pool.elements_per_page())) {
    std::cerr << "page pointers do not match slot offsets\n";
    return 1;
  }

  if (!ExpectThrows([&pool]() { static_cast<void>(pool.allocate_page()); })) {
    std::cerr << "pool exhaustion did not raise an error\n";
    return 1;
  }

  pool.free_page(second);
  if (pool.is_allocated(second)) {
    std::cerr << "freed page is still marked allocated\n";
    return 1;
  }

  const auto recycled = pool.allocate_page();
  if (recycled != second) {
    std::cerr << "freed page was not reused first\n";
    return 1;
  }

  if (!ExpectThrows([&pool]() { pool.free_page(7); })) {
    std::cerr << "invalid page id did not raise an error\n";
    return 1;
  }

  pool.free_page(recycled);
  if (!ExpectThrows([&pool, recycled]() { pool.free_page(recycled); })) {
    std::cerr << "double-free did not raise an error\n";
    return 1;
  }

  pool.reset_pool();
  const auto reset_first = pool.allocate_page();
  const auto reset_second = pool.allocate_page();
  const auto reset_third = pool.allocate_page();
  if (reset_first != 2 || reset_second != 1 || reset_third != 0) {
    std::cerr << "reset pool did not restore the full free list\n";
    return 1;
  }

  std::cout << "page pool tests passed\n";
  return 0;
}
