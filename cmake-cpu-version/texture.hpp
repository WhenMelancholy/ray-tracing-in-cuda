#pragma once

#include "rtweekend.h"
#include "color.h"

class texture {
public:
    virtual color value(double u, double v, const point3 &p) const = 0;
};

class solid_color : public texture {
public:
    solid_color() {}

    solid_color(color c) : color_value(c) {}

    solid_color(double red, double green, double blue) : solid_color(color(red, green, blue)) {}

    virtual color value(double u, double b, const vec3 &p) const override {
        return color_value;
    }

private:
    color color_value;
};

class checker_texture : public texture {
public:
    checker_texture() {}

    checker_texture(std::shared_ptr<texture> _even, std::shared_ptr<texture> _odd)
            : even(_even), odd(_odd) {}

    checker_texture(color c1, color c2) : even(std::make_shared<solid_color>(c1)),
                                          odd(std::make_shared<solid_color>(c2)) {}

    virtual color value(double u, double v, const point3 &p) const override {
        auto sines = sin(10 * p.x()) * sin(10 * p.y()) * sin(10 * p.z());

        if (sines < 0)
            return odd->value(u, v, p);
        else
            return even->value(u, v, p);
    }

public:
    std::shared_ptr<texture> odd;
    std::shared_ptr<texture> even;
};