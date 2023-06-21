#pragma once

#include "hittable.cuh"
#include "vec3.cuh"

#include <cuda.h>

// FIX ���� device_vector �޷��� device ������ʹ�ã�û�ҵ�����������˸���ָ��ʵ��
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

//    __device__ virtual ~hittable_list() override {
//        for (int i = 0; i < len; ++i) {
//            delete objects[i];
//        }
//        delete[] objects;
//    }
};

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
        auto phi = atan2(-p.z(), p.x()) + pi;

        u = phi / (2 * pi);
        v = theta / pi;
    }
};

class xy_rect : public hittable {
public:
    __device__ xy_rect() {}

    __device__ xy_rect(float _x0, float _x1, float _y0, float _y1, float _k, material *mat)
            : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mp(mat) {};

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

//    __device__ virtual ~xy_rect() override {
//        delete mp;
//    }

public:
    material *mp;
    float x0, x1, y0, y1, k;
};