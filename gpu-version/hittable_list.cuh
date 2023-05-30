#pragma once

#include "hittable.cuh"
#include "sphere.cuh"
#include <memory>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
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
