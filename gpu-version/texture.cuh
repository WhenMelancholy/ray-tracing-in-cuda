#pragma once

#include "rtweekend.cuh"
#include "color.cuh"
#include <cuda.h>

class mytexture {
public:
    __device__ virtual color value(float u, float v, const point3 &p) const = 0;
};

class solid_color : public mytexture {
public:
    __device__ __host__ solid_color() {}

    __device__ __host__ solid_color(color c) : color_value(c) {}

    __device__ __host__ solid_color(float red, float green, float blue) : solid_color(color(red, green, blue)) {}

    __device__ virtual color value(float u, float b, const vec3 &p) const override {
        return color_value;
    }

private:
    color color_value;
};

class checker_texture : public mytexture {
public:
    __device__ __host__ checker_texture() {}

    __device__ __host__ checker_texture(mytexture *even, mytexture *odd)
            : even(even), odd(odd) {}

    __device__ checker_texture(color c1, color c2) : even(new solid_color(c1)),
                                                     odd(new solid_color(c2)) {}

    __device__ virtual color value(float u, float v, const point3 &p) const override {
        auto sines = sin(10 * p.x()) * sin(10 * p.y()) * sin(10 * p.z());

        if (sines < 0)
            return odd->value(u, v, p);
        else
            return even->value(u, v, p);
    }

public:
    mytexture *odd;
    mytexture *even;
};