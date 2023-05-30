#pragma once

#include "hittable.cuh"
#include "vec3.cuh"

#include <cuda.h>

class sphere : public hittable {
public:
    __device__ sphere() {}

    __device__ sphere(point3 cen, float r, material *m)
            : center(cen), radius(r), mat_ptr(m) {
    }

    __device__ virtual bool hit(const ray &r, float t_min, float t_max,
                                hit_record &rec) const override;

public:
    point3 center;
    float radius;
    material *mat_ptr;
};

__device__ bool sphere::hit(const ray &r, float t_min, float t_max,
                            hit_record &rec) const {
    vec3 oc = r.origin() - center;
    auto a = r.direction().length_squared();
    auto hb = dot(oc, r.direction());
    auto c = oc.length_squared() - radius * radius;

    auto discriminant = hb * hb - a * c;
    if (discriminant < 0)
        return false;
    auto sqrtd = sqrt(discriminant);

    // find the root that lies in the range;
    auto root = (-hb - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-hb + sqrtd) / a;
        if (root < t_min || t_max < root)
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 normal = (rec.p - center) / radius;
    rec.set_face_normal(r, normal);
    rec.mat_ptr = mat_ptr;

    return true;
}
