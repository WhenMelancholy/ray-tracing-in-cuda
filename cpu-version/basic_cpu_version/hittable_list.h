#pragma once
#include "hittable.h"
#include "sphere.h"
#include <memory>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

class hittable_list : public hittable {
public:
    thrust::host_vector<sphere> objects;
    hittable_list() {}
    hittable_list(sphere* object) {
        add(object);
    }

    void clear() {
        objects.clear();
    }
    void add(sphere* object) { objects.push_back(*object); }

    virtual bool hit(const ray& r, double t_min, double t_max,
        hit_record& rec) const override;
};

bool hittable_list::hit(const ray& r, double t_min, double t_max,
    hit_record& rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    auto closet_so_far = t_max;
    for (const auto& object : objects) {
        if (object.hit(r, t_min, closet_so_far, temp_rec)) {
            hit_anything = true;
            closet_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}
