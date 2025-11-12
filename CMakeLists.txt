cmake_minimum_required(VERSION 3.22)
project(cuda_ops LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_ARCH "sm_86")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=${CUDA_ARCH} -lineinfo --expt-relaxed-constexpr")

function(add_op target_name folder)
    file(GLOB_RECURSE SOURCES "${folder}/*.cpp")
    # Force all sources to be compiled with CUDA
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)
    add_executable(${target_name} ${SOURCES})
    target_include_directories(${target_name} PRIVATE
        "cutlass/include"       # -Icutlass/include
        "cutlass/tools/util/include"  # -Icutlass/tools/util/include
    )
    set_target_properties(${target_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
endfunction()

add_op(elementwise csrc/elementwise)
add_op(sgemm csrc/sgemm)
