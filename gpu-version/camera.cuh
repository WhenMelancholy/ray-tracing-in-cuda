#pragma once

#include "rtweekend.cuh"
#include "ray.cuh"
#include <cuda.h>

class camera {
public:
    __device__ __host__ camera(point3 lookfrom, point3 lookat, vec3 vup,
                               float vfov, float aspect_ratio, float aperature,
                               float focus_dist) {
        float theta = degrees_to_radians(vfov);
        float h = tan(theta / 2.0f);
        float viewport_height = 2.0f * h;
        float viewport_width = aspect_ratio * viewport_height;

        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        origin = lookfrom;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner =
            origin - horizontal / 2 - vertical / 2 - focus_dist * w;

        lens_radius = aperature / 2;
    }

    // UPDATE camera 类也需要随机数生成器
    __device__ ray get_ray(float s, float t, curandState *rng) const {
        // UPDATE 禁用模糊选项
        vec3 rd(0, 0, 0);
        //        vec3 rd = lens_radius * random_in_unit_disk(rng);
        vec3 offset = u * rd.x() + v * rd.y();

        return ray(origin + offset, lower_left_corner + s * horizontal +
                                        t * vertical - origin - offset);
    }

public:
    point3 origin;
    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lens_radius;
};
