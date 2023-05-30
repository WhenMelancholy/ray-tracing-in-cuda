#pragma once

#include "vec3.h"

#include <iostream>

void write_color(std::ostream& out, color c)
{
    out << static_cast<int>(255.99 * c.x()) << ' '
        << static_cast<int>(255.99 * c.y()) << ' '
        << static_cast<int>(255.99 * c.z()) << '\n';
}

void write_color(std::ostream& out, color c, int samples_per_pixel,
    bool gamma_correction = true)
{
    auto r = c.x();
    auto g = c.y();
    auto b = c.z();

    auto scale = 1.0 / samples_per_pixel;
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
