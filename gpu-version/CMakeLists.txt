cmake_minimum_required(VERSION 3.20)
project(parallel_compute CUDA CXX C)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
add_compile_definitions(BASIC_CUDA_VERSION)

add_executable(parallel_compute
        main.cu
        camera.cuh
        color.cuh
        hittable.cuh
        hittable_list.cuh
        material.cuh
        ray.cuh
        rtweekend.cuh
        sphere.cuh
        vec3.cuh
        texture.cuh)

set_target_properties(parallel_compute PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
