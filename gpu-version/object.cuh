#pragma once

#include "hittable.cuh"
#include "vec3.cuh"

#include <assert.h>
#include <cuda.h>

// FIX 由于 device_vector 无法在 device
// 函数中使用（没找到方法），因此改用指针实现
class hittable_list : public hittable {
public:
    int len;
    hittable **objects;

    __device__ __host__ hittable_list() : hittable(class_type::hittable_list) {}

    __device__ __host__ hittable_list(hittable **o, int l) : hittable_list() {
        objects = o;
        len = l;
    }

    __device__ virtual bool hit(const ray &r, float t_min, float t_max,
                                hit_record &rec) const override {
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

    __device__ virtual bool bounding_box(aabb &output_box) const override {
        if (len < 1)
            return false;
        aabb temp_box;
        bool first_true = objects[0]->bounding_box(temp_box);
        if (!first_true)
            return false;
        else
            output_box = temp_box;
        for (int i = 1; i < len; ++i) {
            if (objects[i]->bounding_box(temp_box))
                output_box = surrounding_box(output_box, temp_box);
            else
                return false;
        }
        return true;
    }
};

class sphere : public hittable {
public:
    __device__ __host__ sphere() : hittable(class_type::sphere) {}

    __device__ __host__ sphere(point3 cen, float r, material *m)
        : center(cen), radius(r), mat_ptr(m), hittable(class_type::sphere) {}

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

    __device__ virtual bool bounding_box(aabb &output_box) const override {
        output_box = aabb(center - vec3(radius, radius, radius),
                          center + vec3(radius, radius, radius));
        return true;
    }

    //    __device__ virtual ~sphere() override {
    //        delete mat_ptr;
    //    }

public:
    point3 center;
    float radius;
    material *mat_ptr;

private:
    __device__ static void get_sphere_uv(const point3 &p, float &u, float &v) {
        auto theta = acos(-p.y());
        auto phi = atan2(-p.z(), p.x()) + M_PI;

        u = phi / (2 * M_PI);
        v = theta / M_PI;
    }
};

class xy_rect : public hittable {
public:
    __device__ __host__ xy_rect() : hittable(class_type::xy_rect) {}

    __device__ __host__ xy_rect(float _x0, float _x1, float _y0, float _y1,
                                float _k, material *mat)
        : hittable(class_type::xy_rect), x0(_x0), x1(_x1), y0(_y0), y1(_y1),
          k(_k), mp(mat){};

    __device__ virtual bool hit(const ray &r, float t_min, float t_max,
                                hit_record &rec) const override {
        auto t = (k - r.origin().z()) / r.direction().z();
        if (t < t_min || t > t_max)
            return false;
        auto x = r.origin().x() + t * r.direction().x();
        auto y = r.origin().y() + t * r.direction().y();
        if (x < x0 || x > x1 || y < y0 || y > y1)
            return false;
        rec.u = (x - x0) / (x1 - x0);
        rec.v = (y - y0) / (y1 - y0);
        rec.t = t;
        auto outward_normal = vec3(0, 0, 1);
        rec.set_face_normal(r, outward_normal);
        rec.mat_ptr = mp;
        rec.p = r.at(t);
        return true;
    }

    __device__ virtual bool bounding_box(aabb &output_box) const override {
        // The bounding box must have non-zero width in each dimension, so pad
        // the Z dimension a small amount.
        output_box =
            aabb(point3(x0, y0, k - 0.0001), point3(x1, y1, k + 0.0001));
        return true;
    }

    //    __device__ virtual ~xy_rect() override {
    //        delete mp;
    //    }

public:
    material *mp;
    float x0, x1, y0, y1, k;
};

class xz_rect : public hittable {
public:
    __device__ __host__ xz_rect() : hittable(class_type::xz_rect){};

    __device__ __host__ xz_rect(float _x0, float _x1, float _z0, float _z1,
                                float _k, material *mat)
        : hittable(class_type::xz_rect), x0(_x0), x1(_x1), z0(_z0), z1(_z1),
          k(_k), mp(mat){};

    __device__ virtual bool hit(const ray &r, float t_min, float t_max,
                                hit_record &rec) const override {
        auto t = (k - r.origin().y()) / r.direction().y();
        if (t < t_min || t > t_max)
            return false;
        auto x = r.origin().x() + t * r.direction().x();
        auto z = r.origin().z() + t * r.direction().z();
        if (x < x0 || x > x1 || z < z0 || z > z1)
            return false;
        rec.u = (x - x0) / (x1 - x0);
        rec.v = (z - z0) / (z1 - z0);
        rec.t = t;
        auto outward_normal = vec3(0, 1, 0);
        rec.set_face_normal(r, outward_normal);
        rec.mat_ptr = mp;
        rec.p = r.at(t);
        return true;
    }

    __device__ virtual bool bounding_box(aabb &output_box) const override {
        // The bounding box must have non-zero width in each dimension, so pad
        // the Y dimension a small amount.
        output_box =
            aabb(point3(x0, k - 0.0001, z0), point3(x1, k + 0.0001, z1));
        return true;
    }

public:
    material *mp{};
    float x0{}, x1{}, z0{}, z1{}, k{};
};

class yz_rect : public hittable {
public:
    __device__ __host__ yz_rect() : hittable(class_type::yz_rect){};

    __device__ __host__ yz_rect(float _y0, float _y1, float _z0, float _z1,
                                float _k, material *mat)
        : hittable(class_type::yz_rect), y0(_y0), y1(_y1), z0(_z0), z1(_z1),
          k(_k), mp(mat){};

    __device__ virtual bool hit(const ray &r, float t_min, float t_max,
                                hit_record &rec) const override {
        auto t = (k - r.origin().x()) / r.direction().x();
        if (t < t_min || t > t_max)
            return false;
        auto y = r.origin().y() + t * r.direction().y();
        auto z = r.origin().z() + t * r.direction().z();
        if (y < y0 || y > y1 || z < z0 || z > z1)
            return false;
        rec.u = (y - y0) / (y1 - y0);
        rec.v = (z - z0) / (z1 - z0);
        rec.t = t;
        auto outward_normal = vec3(1, 0, 0);
        rec.set_face_normal(r, outward_normal);
        rec.mat_ptr = mp;
        rec.p = r.at(t);
        return true;
    }

    __device__ virtual bool bounding_box(aabb &output_box) const override {
        // The bounding box must have non-zero width in each dimension, so pad
        // the X dimension a small amount.
        output_box =
            aabb(point3(k - 0.0001, y0, z0), point3(k + 0.0001, y1, z1));
        return true;
    }

public:
    material *mp{};
    float y0{}, y1{}, z0{}, z1{}, k{};
};

__device__ __host__ bool quadratic(float a, float b, float c, float &t0,
                                   float &t1) {
    float delta = b * b - 4 * a * c;
    if (delta < 0)
        return false;
    float sqrt_delta = sqrt(delta);

    t0 = -0.5f * (b - sqrt_delta) / a;
    t1 = -0.5f * (b + sqrt_delta) / a;
    if (t0 > t1) {
        float temp = t0;
        t0 = t1;
        t1 = temp;
    }
    return true;
}

class cylinder : public hittable {
public:
    __device__ __host__ cylinder() : hittable(class_type::cylinder){};

    __device__ __host__ cylinder(float _radius, float _zmin, float _zmax,
                                 material *mat)
        : hittable(class_type::cylinder), radius(_radius), zmin(_zmin),
          zmax(_zmax), mat_ptr(mat){};

    __device__ __host__ void rotate(vec3 axis, float rad) {
        o2w = ::rotate(axis, rad) * o2w;
    }

    __device__ __host__ void translate(vec3 offset) {
        o2w = ::translate(offset) * o2w;
    }

    __device__ virtual bool hit(const ray &r, float t_min, float t_max,
                                hit_record &rec) const override {
        auto object_ray = o2w.inverse().apply_ray(r);

        auto dx = object_ray.direction().x(), dy = object_ray.direction().y(),
             dz = object_ray.direction().z();
        auto ox = object_ray.origin().x(), oy = object_ray.origin().y(),
             oz = object_ray.origin().z();
        // solve quadratic equation for t values
        float a = dx * dx + dy * dy;
        float b = 2 * (dx * ox + dy * oy);
        float c = ox * ox + oy * oy - radius * radius;
        float t0, t1;
        if (!quadratic(a, b, c, t0, t1))
            return false;
        if (t0 > t1) {
            float temp = t0;
            t0 = t1;
            t1 = temp;
        }
        if (t0 > t_max || t1 < t_min)
            return false;
        float t = t0;
        if (t0 < t_min) {
            t = t1;
            if (t > t_max)
                return false;
        }
        // compute sphere hit position and phi and fill hit record
        auto object_p = object_ray.at(t);
        if (object_p.z() < zmin || object_p.z() > zmax) {
            if (t == t1)
                return false;
            t = t1;
            if (t > t_max || t < t_min)
                return false;
            object_p = object_ray.at(t);
            if (object_p.z() < zmin || object_p.z() > zmax)
                return false;
        }
        auto object_normal = vec3(object_p.x(), object_p.y(), 0).normalize();

        auto world_p = o2w.apply_point(object_p);
        auto world_normal = o2w.apply_normal(object_normal);

        rec.p = world_p;
        rec.set_face_normal(r, world_normal);
        rec.mat_ptr = mat_ptr;
        rec.t = t;

        float phi = atan2(object_p.y(), object_p.x());
        phi += 2 * M_PI;
        assert(0 <= phi && phi <= 4 * M_PI);

        rec.u = phi / (4 * M_PI);
        rec.v = (object_p.z() - zmin) / (zmax - zmin);
        return true;
    }

    __device__ virtual bool bounding_box(aabb &output_box) const override {
        output_box =
            aabb(point3(-radius, -radius, zmin), point3(radius, radius, zmax));
        return true;
    }

public:
    material *mat_ptr{};
    float radius{};
    float zmin{}, zmax{};
    transform o2w{identity()};
};

__device__ bool box_compare(hittable *a, hittable *b, int axis) {
    aabb box_a, box_b;
    if (!a->bounding_box(box_a) || !b->bounding_box(box_b))
        assert(false && "no bounding box in bvh_node constructor");
    return box_a.min().e[axis] < box_b.min().e[axis];
}

__device__ void pop_sort(hittable **objects, int start, int end, int axis) {
    for (int i = start; i < end - 1; ++i) {
        for (int j = i + 1; j < end; ++j) {
            if (box_compare(objects[i], objects[j], axis)) {
                auto temp = objects[i];
                objects[i] = objects[j];
                objects[j] = temp;
            }
        }
    }
}

class bvh_node : public hittable {
public:
    __device__ bvh_node() : hittable(class_type::bvh_node) {}
    __device__ bvh_node(const hittable_list &list)
        : bvh_node(list.objects, 0, list.len) {}
    __device__ bvh_node(hittable **src_objects, int start, int end)
        : hittable(class_type::bvh_node) {
        int axis = (start - end) % 3;
        auto obj_count = end - start;
        if (obj_count == 1) {
            left = right = src_objects[start];
        } else if (obj_count == 2) {
            left = src_objects[start];
            right = src_objects[start + 1];
        } else {
            pop_sort(src_objects, start, end, axis);
            auto mid = (start + end) / 2;
            left = new bvh_node(src_objects, start, mid);
            right = new bvh_node(src_objects, mid, end);
        }

        aabb box_left, box_right;
        if (!left->bounding_box(box_left) || !right->bounding_box(box_right))
            assert(false && "no bounding box in bvh_node constructor");
        box = surrounding_box(box_left, box_right);
    }

    __device__ virtual bool bounding_box(aabb &output_box) const override {
        output_box = box;
        return true;
    }

    __device__ bool hit(const ray &r, float t_min, float t_max,
                        hit_record &rec) const override {
        if (!box.hit(r, t_min, t_max))
            return false;
        bool hit_left = left->hit(r, t_min, t_max, rec);
        bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);
        return hit_left || hit_right;
    }

public:
    hittable *left;
    hittable *right;
    aabb box;
};