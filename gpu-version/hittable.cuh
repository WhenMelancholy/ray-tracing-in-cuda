#pragma once

#include "ray.cuh"
#include "rtweekend.cuh"
#include "color.cuh"

#include <cuda.h>

class material;

struct hit_record {
    point3 p;
    vec3 normal;
    material *mat_ptr;
    float t;
    float u;
    float v;
    bool front_face;

    __device__ void set_face_normal(const ray &r, const vec3 &outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hittable {
public:
    __device__ virtual bool hit(const ray &r, float t_min, float t_max,
                                hit_record &rec) const = 0;

//    __device__ virtual ~hittable() {}
};
