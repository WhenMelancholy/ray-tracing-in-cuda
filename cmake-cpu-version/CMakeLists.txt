cmake_minimum_required(VERSION 3.24)
project(cmake_cpu_version)

set(CMAKE_CXX_STANDARD 23)

FIND_PACKAGE(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

add_executable(cmake_cpu_version
        main.cpp
        camera.h
        color.h
        hittable.h
        hittable_list.h
        material.h ray.h
        rtweekend.h
        sphere.h
        vec3.h
        texture.hpp)
