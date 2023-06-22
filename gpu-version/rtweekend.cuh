#pragma once

#include <cstdlib>
#include <limits>
#include <iostream>

#include <cuda.h>
#include <curand_kernel.h>

//__constant__ float inf;
//__constant__ float pi;

#define pi 3.14159265f

__device__ __host__ float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0f;
}

// UPDATE 应该使用 uniform，normal 是正态分布
// UPDATE 在设备端中需要使用 curand 库生成随机数
__device__ float random_float(curandState *state) {
    return curand_uniform(state);
}

__device__ float random_float(float min, float max, curandState *state) {
    return min + (max - min) * random_float(state);
}

__device__ __host__ float clamp(float x, float min, float max) {
    if (x < min)
        return min;
    if (x > max)
        return max;
    return x;
}

// debug 输出函数
#define when(...) fprintf(stderr,__VA_ARGS__)
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                  file << ":" << line << " '" << func << "' \n";
        std::cerr << "Error string: " << cudaGetErrorString(result) << "\n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}