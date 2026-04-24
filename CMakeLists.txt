cmake_minimum_required(VERSION 3.24)

project(dynamic_sparse_decode LANGUAGES CXX)

include(CheckLanguage)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(DSD_ENABLE_CUDA "Build CUDA targets when a CUDA toolkit is available" ON)
set(DSD_CUDA_ARCHITECTURES "90" CACHE STRING "CUDA architectures for device code")

set(DSD_CORE_SOURCES
  src/page_pool.cpp
  src/paged_kv_cache.cpp
  src/reference_kernels.cpp
  src/decode_pipeline.cpp
  src/synthetic_data.cpp
  src/continuous_batching.cpp
)

check_language(CUDA)
set(DSD_HAVE_CUDA OFF)
if(DSD_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
  set(DSD_HAVE_CUDA ON)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  get_filename_component(DSD_CUDA_BIN_DIR "${CMAKE_CUDA_COMPILER}" DIRECTORY)
  get_filename_component(DSD_CUDA_ROOT_DIR "${DSD_CUDA_BIN_DIR}/.." ABSOLUTE)
  set(DSD_CUDA_INCLUDE_DIR "${DSD_CUDA_ROOT_DIR}/include")
  set(DSD_CUDA_LIBRARY_DIR "${DSD_CUDA_ROOT_DIR}/lib64")
  set(DSD_CUDA_TARGET_LIBRARY_DIR "${DSD_CUDA_ROOT_DIR}/targets/x86_64-linux/lib")
  if(NOT EXISTS "${DSD_CUDA_INCLUDE_DIR}/cuda_runtime.h")
    find_path(DSD_CUDA_INCLUDE_DIR
      NAMES cuda_runtime.h
      HINTS
        "${CUDAToolkit_TARGET_DIR}/include"
        "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/include"
        "$ENV{CUDA_HOME}/include"
        "$ENV{CUDA_PATH}/include")
  endif()
  set(CMAKE_CUDA_ARCHITECTURES ${DSD_CUDA_ARCHITECTURES})
else()
  list(APPEND DSD_CORE_SOURCES src/cuda_dense_attention_stub.cpp)
  list(APPEND DSD_CORE_SOURCES src/cuda_sparse_attention_stub.cpp)
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

add_executable(dsd_continuous_bench benchmarks/bench_continuous_decode.cpp)
target_link_libraries(dsd_continuous_bench PRIVATE dsd_core)

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
    src/cuda/sparse_batched_attention.cu
  )

  set_source_files_properties(src/device_page_pool.cpp PROPERTIES LANGUAGE CUDA)

  set_target_properties(dsd_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_RUNTIME_LIBRARY Shared
  )

  target_include_directories(dsd_cuda PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${DSD_CUDA_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_include_directories(dsd_core PUBLIC
    ${DSD_CUDA_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_compile_options(dsd_core PUBLIC
    "$<$<COMPILE_LANGUAGE:CXX>:-I${DSD_CUDA_INCLUDE_DIR}>")
  target_link_directories(dsd_cuda PUBLIC
    ${DSD_CUDA_LIBRARY_DIR}
    ${DSD_CUDA_TARGET_LIBRARY_DIR})
  target_link_options(dsd_core PUBLIC
    "-L${DSD_CUDA_LIBRARY_DIR}"
    "-L${DSD_CUDA_TARGET_LIBRARY_DIR}")
  target_compile_definitions(dsd_cuda PUBLIC DSD_HAVE_CUDA=1)
  target_link_libraries(dsd_cuda PUBLIC
    ${DSD_CUDA_LIBRARY_DIR}/libcudart.so
    ${DSD_CUDA_LIBRARY_DIR}/libcudadevrt.a
    ${CMAKE_DL_LIBS})
  target_link_libraries(dsd_core PUBLIC dsd_cuda)

  add_executable(dsd_device_page_pool_test tests/test_device_page_pool.cpp)
  target_link_libraries(dsd_device_page_pool_test PRIVATE dsd_core)
  add_test(NAME dsd_device_page_pool_test COMMAND dsd_device_page_pool_test)

  add_executable(dsd_dense_cuda_test tests/test_dense_cuda.cpp)
  target_link_libraries(dsd_dense_cuda_test PRIVATE dsd_core)
  add_test(NAME dsd_dense_cuda_test COMMAND dsd_dense_cuda_test)

  add_executable(dsd_sparse_cuda_test tests/test_sparse_cuda.cpp)
  target_link_libraries(dsd_sparse_cuda_test PRIVATE dsd_core)
  add_test(NAME dsd_sparse_cuda_test COMMAND dsd_sparse_cuda_test)

  add_executable(dsd_continuous_batching_test tests/test_continuous_batching.cpp)
  target_link_libraries(dsd_continuous_batching_test PRIVATE dsd_core)
  add_test(NAME dsd_continuous_batching_test COMMAND dsd_continuous_batching_test)

  set(DSD_CUDA_BUILD_RPATH "$<TARGET_FILE_DIR:CUDA::cudart>")
  set_target_properties(
    dsd_demo
    dsd_bench
    dsd_continuous_bench
    dsd_page_pool_test
    dsd_reference_test
    dsd_device_page_pool_test
    dsd_dense_cuda_test
    dsd_sparse_cuda_test
    dsd_continuous_batching_test
    PROPERTIES BUILD_RPATH "${DSD_CUDA_BUILD_RPATH}")
else()
  target_compile_definitions(dsd_core PUBLIC DSD_HAVE_CUDA=0)
endif()
