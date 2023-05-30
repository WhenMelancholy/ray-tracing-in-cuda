#pragma once

#include "vec3.cuh"

#include <cuda.h>

class ray {
public:
    __device__ ray() {}

    __device__ ray(const point3 &origin, const vec3 &dir) : orig(origin), dir(dir) {}

    __device__ point3 origin() const { return orig; }

    __device__ vec3 direction() const { return dir; }

    __device__ point3 at(float t) const { return orig + t * dir; }

public:
    point3 orig;
    vec3 dir;
};
