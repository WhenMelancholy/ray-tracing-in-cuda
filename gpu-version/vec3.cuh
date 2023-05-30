#pragma once

#include <iostream>
#include "rtweekend.cuh"

#include <cuda.h>
#include <curand_kernel.h>

class vec3 {
public:
    // UPDATE 这个文件中的随机数相关代码均需要删除，使用 curand 库重新实现
    // 一个十分神奇的编译问题，假如删掉了下面这个空行，就会导致编译错误
    // 根据测试，可能是行末换行符+文件编码共同引起的问题
    static __device__ vec3 random(curandState *state) {
        return {random_float(state), random_float(state), random_float(state)};
    }

    static __device__ vec3 random(float min, float max, curandState *state) {
        return {random_float(min, max, state), random_float(min, max, state),
                random_float(min, max, state)};
    }


public:
    __host__ __device__ vec3()
            : e{0, 0, 0} {
    }

    __host__ __device__ vec3(float e0, float e1, float e2)
            : e{e0, e1, e2} {
    }

    __host__ __device__ float x() const { return e[0]; }

    __host__ __device__ float y() const { return e[1]; }

    __host__ __device__ float z() const { return e[2]; }

    __host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }

    __host__ __device__ float operator[](int i) const { return e[i]; }

    __host__ __device__ float &operator[](int i) { return e[i]; }

    __host__ __device__ vec3 &operator+=(const vec3 &v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __host__ __device__ vec3 &operator*=(const float t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ vec3 &operator/=(const float t) { return *this *= 1 / t; }

    __host__ __device__ float length() const { return sqrt(length_squared()); }

    __host__ __device__ float length_squared() const {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    __host__ __device__ bool near_zero(const float &eps = 1e-8) {
        return fabs(e[0]) < eps && fabs(e[1]) < eps && fabs(e[2]) < eps;
    }

public:
    float e[3];
};

using point3 = vec3;
using color = vec3;

// begin utility functions
std::ostream &operator<<(std::ostream &out, const vec3 &v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ vec3 operator+(const vec3 &u, const vec3 &v) {
    return vec3(u.x() + v.x(), u.y() + v.y(), u.z() + v.z());
}

__host__ __device__ vec3 operator-(const vec3 &u, const vec3 &v) {
    return vec3(u.x() - v.x(), u.y() - v.y(), u.z() - v.z());
}

__host__ __device__ vec3 operator*(const vec3 &u, const vec3 &v) {
    return vec3(u.x() * v.x(), u.y() * v.y(), u.z() * v.z());
}

__host__ __device__ vec3 operator*(float t, const vec3 &v) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ vec3 operator*(const vec3 &v, float t) { return t * v; }

__host__ __device__ vec3 operator/(vec3 v, float t) { return (1 / t) * v; }

__host__ __device__ float dot(const vec3 &u, const vec3 &v) {
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

__host__ __device__ vec3 cross(const vec3 &u, const vec3 &v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ vec3 unit_vector(vec3 v) { return v / v.length(); }

__host__ __device__ vec3 reflect(const vec3 &v, const vec3 &n) {
    return v - 2 * dot(v, n) * n;
}

__host__ __device__ vec3 refract(const vec3 &uv, const vec3 &n, float etai_over_etat) {
    auto cos_theta = min(dot(-uv, n), 1.0f);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -sqrt(fabs(1.0f - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

__device__ vec3 random_in_unit_sphere(curandState *state) {
    for (;;) {
        auto p = vec3(random_float(state), random_float(state), random_float(state));
        if (p.length_squared() >= 1)
            continue;
        return p;
    }
}

__device__ vec3 random_unit_vector(curandState *state) {
    return unit_vector(random_in_unit_sphere(state));
}

__device__ vec3 random_in_unit_disk(curandState *state) {
    while (true) {
        auto p = vec3(random_float(-1, 1, state), random_float(-1, 1, state), 0);
        if (p.length_squared() >= 1)
            continue;
        return p;
    }
//    auto p = vec3(random_float(-1, 1, state), random_float(-1, 1, state), 0);
//    return random_float(state) * unit_vector(p);
}