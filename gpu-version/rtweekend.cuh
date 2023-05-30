#pragma once

#include <cstdlib>
#include <limits>

#include <cuda.h>
#include <curand_kernel.h>

//__constant__ float inf;
//__constant__ float pi;

#define pi 3.14159265f

__device__ float degrees_to_radians(float degrees) {
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