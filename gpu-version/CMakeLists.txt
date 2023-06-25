cmake_minimum_required(VERSION 3.20)
project(ray_tracing CUDA CXX C)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
add_compile_definitions(BASIC_CUDA_VERSION)

find_package(nlohmann_json CONFIG REQUIRED)
find_package(Stb REQUIRED)

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

add_executable(parallel_compute main.cu ${COMMON_HEADERS})
add_executable(parser parser.cu ${COMMON_HEADERS})

set_target_properties(parallel_compute PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(parallel_compute PRIVATE ${Stb_INCLUDE_DIR})
target_link_libraries(parallel_compute nlohmann_json::nlohmann_json)
# target_compile_options(parallel_compute PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
# target_compile_options(parallel_compute PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g>)
# target_compile_options(parallel_compute PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -rdynamic>)
# target_compile_options(parallel_compute PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=64>)

set_target_properties(parser PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(parser nlohmann_json::nlohmann_json)
target_include_directories(parser PRIVATE ${Stb_INCLUDE_DIR})


configure_file(sample_scene.json sample_scene.json COPYONLY)
configure_file(basic_scene.json basic_scene.json COPYONLY)