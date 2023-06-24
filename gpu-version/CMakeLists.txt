cmake_minimum_required(VERSION 3.20)
project(ray_tracing CUDA CXX C)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
add_compile_definitions(BASIC_CUDA_VERSION)

find_package(nlohmann_json CONFIG REQUIRED)

set(COMMON_HEADERS
        camera.cuh
        color.cuh
        hittable.cuh
        object.cuh
        material.cuh
        ray.cuh
        rtweekend.cuh
        vec3.cuh
        texture.cuh
        parser.hpp
        )

add_compile_options("$<$<CONFIG:DEBUG>:-g;-G>")
#add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>")

add_executable(parallel_compute main.cu ${COMMON_HEADERS})
add_executable(parser parser.cu ${COMMON_HEADERS})

set_target_properties(parallel_compute PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(parallel_compute nlohmann_json::nlohmann_json)

set_target_properties(parser PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(parser nlohmann_json::nlohmann_json)

configure_file(sample_scene.json sample_scene.json COPYONLY)