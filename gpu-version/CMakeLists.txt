cmake_minimum_required(VERSION 3.20)
project(parallel_compute CUDA CXX C)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
add_compile_definitions(BASIC_CUDA_VERSION)
#set(CMAKE_CUDA_COMPILER "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.5\\bin\\nvcc.exe")

add_executable(parallel_compute main.cu camera.cuh color.cuh hittable.cuh hittable_list.cuh material.cuh ray.cuh rtweekend.cuh sphere.cuh vec3.cuh)

set_target_properties(parallel_compute PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
