cmake_minimum_required(VERSION 3.24)

project(dynamic_sparse_decode LANGUAGES CXX)

include(CheckLanguage)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(DSD_ENABLE_CUDA "Build CUDA targets when a CUDA toolkit is available" ON)
set(DSD_CUDA_ARCHITECTURES "90" CACHE STRING "CUDA architectures for device code")

set(DSD_CORE_SOURCES
  src/page_pool.cpp
  src/paged_kv_cache.cpp
  src/reference_kernels.cpp
  src/decode_pipeline.cpp
  src/synthetic_data.cpp
)

check_language(CUDA)
set(DSD_HAVE_CUDA OFF)
if(DSD_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
  set(DSD_HAVE_CUDA ON)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_ARCHITECTURES ${DSD_CUDA_ARCHITECTURES})
else()
  list(APPEND DSD_CORE_SOURCES src/cuda_dense_attention_stub.cpp)
endif()

add_library(dsd_core
  ${DSD_CORE_SOURCES}
)

target_include_directories(dsd_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_options(dsd_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(dsd_demo src/main.cpp)
target_link_libraries(dsd_demo PRIVATE dsd_core)

add_executable(dsd_bench benchmarks/bench_decode.cpp)
target_link_libraries(dsd_bench PRIVATE dsd_core)

enable_testing()

add_executable(dsd_page_pool_test tests/test_page_pool.cpp)
target_link_libraries(dsd_page_pool_test PRIVATE dsd_core)
add_test(NAME dsd_page_pool_test COMMAND dsd_page_pool_test)

add_executable(dsd_reference_test tests/test_reference.cpp)
target_link_libraries(dsd_reference_test PRIVATE dsd_core)
add_test(NAME dsd_reference_test COMMAND dsd_reference_test)

if(DSD_HAVE_CUDA)
  target_compile_definitions(dsd_core PUBLIC DSD_HAVE_CUDA=1)

  add_library(dsd_cuda
    src/device_page_pool.cpp
    src/cuda/kernels.cu
  )

  set_target_properties(dsd_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
  )

  target_include_directories(dsd_cuda PUBLIC ${PROJECT_SOURCE_DIR}/include)
  target_compile_definitions(dsd_cuda PUBLIC DSD_HAVE_CUDA=1)
  target_link_libraries(dsd_cuda PUBLIC CUDA::cudart)
  target_link_libraries(dsd_core PUBLIC dsd_cuda)

  add_executable(dsd_device_page_pool_test tests/test_device_page_pool.cpp)
  target_link_libraries(dsd_device_page_pool_test PRIVATE dsd_core)
  add_test(NAME dsd_device_page_pool_test COMMAND dsd_device_page_pool_test)

  add_executable(dsd_dense_cuda_test tests/test_dense_cuda.cpp)
  target_link_libraries(dsd_dense_cuda_test PRIVATE dsd_core)
  add_test(NAME dsd_dense_cuda_test COMMAND dsd_dense_cuda_test)

  set(DSD_CUDA_BUILD_RPATH "$<TARGET_FILE_DIR:CUDA::cudart>")
  set_target_properties(
    dsd_demo
    dsd_bench
    dsd_page_pool_test
    dsd_reference_test
    dsd_device_page_pool_test
    dsd_dense_cuda_test
    PROPERTIES BUILD_RPATH "${DSD_CUDA_BUILD_RPATH}")
else()
  target_compile_definitions(dsd_core PUBLIC DSD_HAVE_CUDA=0)
endif()
