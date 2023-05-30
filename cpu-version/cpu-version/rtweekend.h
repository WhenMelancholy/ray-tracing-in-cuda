#pragma once
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>

using std::make_shared;
using std::shared_ptr;
using std::sqrt;

const double inf = std::numeric_limits<double>::infinity();
const double pi = acos(-1);

inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

inline double random_double() {
#ifdef MT_RANDOM_GENERATOR
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
#else
    return rand() / double(int(RAND_MAX)+1);
#endif
}

inline double random_double(double min, double max) {
    return min + (max - min) * random_double();
}

inline double clamp(double x, double min, double max) {
    if (x < min)
        return min;
    if (x > max)
        return max;
    return x;
}

#include "ray.h"
#include "vec3.h"
