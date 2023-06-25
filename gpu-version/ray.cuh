#pragma once

#include "vec3.cuh"

#include <cuda.h>

class ray {
public:
    __device__ __host__ ray() {}

    __device__ __host__ ray(const point3 &origin, const vec3 &dir)
        : orig(origin), dir(dir) {}

    __device__ __host__ point3 origin() const { return orig; }

    __device__ __host__ vec3 direction() const { return dir; }

    __device__ __host__ point3 at(float t) const { return orig + t * dir; }

public:
    point3 orig;
    vec3 dir;
};

__device__ __host__ ray transform::apply_ray(const ray &r) const {
    return ray(apply_point(r.origin()), apply_vec(r.direction()));
}