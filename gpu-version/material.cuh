#pragma once

#include "rtweekend.cuh"
#include "texture.cuh"

#include <cuda.h>
#include <curand_kernel.h>

struct hit_record;

class material {
public:
    // 需要使用 curand 库来实现随机数的产生
    __device__ virtual bool scatter(
            const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered,
            curandState *state) const = 0;

    __device__ virtual color emitted(float u, float v, const point3 &p) const {
        return color(0, 0, 0);
    }
};

class lambertian : public material {
public:
    __device__ lambertian(const color &a)
            : albedo(new solid_color(a)) {}

    __device__ lambertian(mytexture *a) : albedo(a) {}

    __device__ virtual bool scatter(
            const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered,
            curandState *state) const override {
        // UPDATE 将 random_in_unit_vector 替换仍然会有渲染错误
        // 但是结果明显不同
        auto scatter_direction = rec.normal + random_in_unit_sphere(state);

        //  UPDATE 尝试删去对微弱光线的特判
        // 并不是此处引起的渲染错误
        if (scatter_direction.near_zero())
            scatter_direction = rec.normal;

        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo->value(rec.u, rec.v, rec.p);
        return true;
    }

public:
    mytexture *albedo;
};

class metal : public material {
public:
    __device__ metal(const color &a, float f)
            : albedo(a), fuzz(f < 1 ? f : 1) {
    }

    __device__ virtual bool scatter(
            const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered,
            curandState *state) const override {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(state));
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0.0f);
    }

public:
    color albedo;
    float fuzz;
};

__device__ bool refract(const vec3 &v, const vec3 &n, float ni_over_nt, vec3 &refracted) {
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant > 0) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    } else
        return false;
}

class dielectric : public material {
public:
    __device__ dielectric(float index_of_refraction)
            : ir(index_of_refraction) {
    }

    __device__ virtual bool scatter(
            const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered,
            curandState *state) const override {
        // UPDATE 测试不同的材质实现会不会带来不同的渲染结果
        if (false) {
            vec3 outward_normal;
            vec3 reflected = reflect(r_in.direction(), rec.normal);
            float ni_over_nt;
            attenuation = vec3(1.0, 1.0, 1.0);
            vec3 refracted;
            float reflect_prob;
            float cosine;
            if (dot(r_in.direction(), rec.normal) > 0.0f) {
                outward_normal = -rec.normal;
                ni_over_nt = ir;
                cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
                cosine = sqrt(1.0f - ir * ir * (1 - cosine * cosine));
            } else {
                outward_normal = rec.normal;
                ni_over_nt = 1.0f / ir;
                cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
            }
            if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
                reflect_prob = reflectance(cosine, ir);
            else
                reflect_prob = 1.0f;
            if (random_float(state) < reflect_prob)
                scattered = ray(rec.p, reflected);
            else
                scattered = ray(rec.p, refracted);
            return true;
        }

        attenuation = color(1.0, 1.0, 1.0);
        float refraction_ratio = rec.front_face ? (1.0f / ir) : ir;

        vec3 unit_direction = unit_vector(r_in.direction());
        float cos_theta = min(dot(-unit_direction, rec.normal), 1.0f);
        float sin_theta = sqrt(1 - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_float(state))
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, refraction_ratio);

        scattered = ray(rec.p, direction);
        return true;
    }

public:
    float ir;

private:
    __device__ static float reflectance(float cosine, float ref_idx) {
        auto r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow(1 - cosine, 5);
    }
};

class diffuse_light : public material {
public:
    __device__ diffuse_light(mytexture *a) : emit(a) {}

    __device__ diffuse_light(color c) : emit(new solid_color(c)) {}

    __device__ virtual bool scatter(
            const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered,
            curandState *state) const override {
        return false;
    }

    __device__ virtual color emitted(float u, float v, const point3 &p) const override {
        return emit->value(u, v, p);
    }

public:
    mytexture *emit;
};