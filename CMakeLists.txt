cmake_minimum_required(VERSION 3.28)
project(cuda-optim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 80)
set(CUDA_ARCH "80" CACHE STRING "CUDA architecture")

find_package(CUDAToolkit REQUIRED)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/debug)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/release)

set(
    GEMM_SRC
    src/gemm/gemm.cu
    src/gemm/kernel/0_naive.cuh
    src/gemm/kernel/1_shared_mem.cuh
    src/gemm/kernel/2_onedim_blocktile.cuh
    src/gemm/kernel/3_twodim_blocktile.cuh
    src/gemm/kernel/4_twodim_blocktile_vectorized.cuh
    src/gemm/kernel/x_cublas.cuh
    src/gemm/kernels.cuh
    src/gemm/timer.cuh
)
set(
    GENERAL_SRC
    src/utils.cuh
    src/arg_parser.h
)
set(
    CUTLASS_INCL
    cutlass/include
    cutlass/tools/util/include
)

add_executable(gemm ${GEMM_SRC} ${GENERAL_SRC})
target_include_directories(gemm PRIVATE ${CUTLASS_INCL} ${WORKING_DIRECTORY}/src)

set_target_properties(gemm PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(gemm PRIVATE --expt-relaxed-constexpr -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH})

set_target_properties(gemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(gemm PRIVATE CUDA::cublas)
set_target_properties(gemm PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")