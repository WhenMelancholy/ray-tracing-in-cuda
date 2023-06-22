cmake_minimum_required(VERSION 3.20)
project(parallel_compute CUDA CXX C)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
add_compile_definitions(BASIC_CUDA_VERSION)

find_package(nlohmann_json CONFIG REQUIRED)

add_executable(parallel_compute
        main.cu
        camera.cuh
        color.cuh
        hittable.cuh
        object.cuh
        material.cuh
        ray.cuh
        rtweekend.cuh
        vec3.cuh
        texture.cuh
        parser.hpp)

set_target_properties(parallel_compute PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(parallel_compute PRIVATE nlohmann_json::nlohmann_json)
