#pragma once

#include "hittable.cuh"
#include "vec3.cuh"

#include <cuda.h>

// FIX 由于 device_vector 无法在 device 函数中使用（没找到方法），因此改用指针实现
class hittable_list : public hittable {
public:
    int len;
    hittable **objects;

    __device__ hittable_list() {}

    __device__ hittable_list(hittable **o, int l) {
        objects = o;
        len = l;
    }

    __device__ virtual bool hit(const ray &r, float t_min, float t_max,
                                hit_record &rec) const override;
};

__device__ bool hittable_list::hit(const ray &r, float t_min, float t_max,
                                   hit_record &rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    auto closet_so_far = t_max;
    for (int i = 0; i < len; ++i) {
        if (objects[i]->hit(r, t_min, closet_so_far, temp_rec)) {
            hit_anything = true;
            closet_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

class sphere : public hittable {
public:
    __device__ sphere() {}

    __device__ sphere(point3 cen, float r, material *m)
            : center(cen), radius(r), mat_ptr(m) {
    }

    __device__ virtual bool hit(const ray &r, float t_min, float t_max,
                                hit_record &rec) const override {
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
        get_sphere_uv(normal, rec.u, rec.v);
        rec.mat_ptr = mat_ptr;

        return true;
    }


public:
    point3 center;
    float radius;
    material *mat_ptr;

private:
    __device__ static void get_sphere_uv(const point3 &p, float &u, float &v) {
        auto theta = acos(-p.y());
        auto phi = atan2(-p.z(), p.x()) + pi;

        u = phi / (2 * pi);
        v = theta / pi;
    }
};