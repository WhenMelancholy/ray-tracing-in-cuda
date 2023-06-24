#pragma once

#include "vec3.cuh"

#include <iostream>
#include <cstdio>
#include <cuda.h>

void write_color(std::ostream &out, color c) {
    out << static_cast<int>(255.99 * c.x()) << ' '
        << static_cast<int>(255.99 * c.y()) << ' '
        << static_cast<int>(255.99 * c.z()) << '\n';
}

void write_color(std::ostream &out, color c, int samples_per_pixel,
                 bool gamma_correction = true) {
    auto r = c.x();
    auto g = c.y();
    auto b = c.z();

    // UPDATE 发现存在 r<0 g<0 b<0 的情况，开始 debug
    if (r < 0 || g < 0 || b < 0) {
        fprintf(stderr, "error color value less than 0\n");
    }

    auto scale = 1.0f / float(samples_per_pixel);
    r *= scale;
    g *= scale;
    b *= scale;

    if (gamma_correction) {
        r = sqrt(r);
        g = sqrt(g);
        b = sqrt(b);
    }

    out << static_cast<int>(256 * clamp(r, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(g, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(b, 0.0, 0.999)) << '\n';
}

void write_color(FILE *file, color c, int samples_per_pixel,
                 bool gamma_correction = true) {
    auto r = c.x();
    auto g = c.y();
    auto b = c.z();

    // UPDATE 发现存在 r<0 g<0 b<0 的情况，开始 debug
    if (r < 0 || g < 0 || b < 0) {
        fprintf(stderr, "error color value less than 0\n");
    }

    auto scale = 1.0f / float(samples_per_pixel);
    r *= scale;
    g *= scale;
    b *= scale;

    if (gamma_correction) {
        r = sqrt(r);
        g = sqrt(g);
        b = sqrt(b);
    }

    fprintf(file, "%d %d %d\n", static_cast<int>(256 * clamp(r, 0.0, 0.999)),
            static_cast<int>(256 * clamp(g, 0.0, 0.999)),
            static_cast<int>(256 * clamp(b, 0.0, 0.999)));
}
