#pragma once
#include "hittable.h"
#include "vec3.h"

class sphere : public hittable {
public:
    sphere() { }
    sphere(point3 cen, double r, material* m)
        : center(cen)
        , radius(r)
        , mat_ptr(m)
    {
    }

    virtual bool hit(const ray& r, double t_min, double t_max,
        hit_record& rec) const override;

public:
    point3 center;
    double radius;
    material* mat_ptr;
};

bool sphere::hit(const ray& r, double t_min, double t_max,
    hit_record& rec) const
{
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
