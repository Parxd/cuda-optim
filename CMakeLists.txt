cmake_minimum_required(VERSION 3.28)
project(cuda-optim LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 80)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --x=cu")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --x=cu" )

# Disable this for non-debug build
add_compile_options(-g -G)

# clangd doesn't understand a lot of nvcc flags, avoid compile_commands.json for now
set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)

set(
    GEMM_SRC
    src/gemm.cc
    src/kernel/0_naive.hh
    src/kernel/1_smem.hh
    src/kernel/2_1dim_threadtile.hh
    src/kernel/3_2dim_threadtile.hh
    src/kernel/4_vectorize.hh
    src/kernel/5_warptile.hh
    src/kernel/x_cublas.hh
    src/kernels.hh
    src/timer.hh
)
set(
    GENERAL_SRC
    src/utils.hh
    src/arg_parser.h
)
set(
    CUTLASS_INCL
    cutlass/include
    cutlass/tools/util/include
)
foreach(file ${GEMM_SRC})
    set_source_files_properties(${file} PROPERTIES LANGUAGE CUDA)
endforeach()

add_executable(gemm ${GEMM_SRC} ${GENERAL_SRC})
target_compile_options(gemm PRIVATE --expt-relaxed-constexpr)
target_link_libraries(gemm PRIVATE -lcublas)
target_include_directories(gemm PRIVATE ${CUTLASS_INCL} ${WORKING_DIRECTORY}/src)
target_include_directories(gemm PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(gemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)