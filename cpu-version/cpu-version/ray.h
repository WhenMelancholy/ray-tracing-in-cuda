#pragma once

#include "vec3.h"

class ray {
public:
	ray() {}
	ray(const point3& origin, const vec3& dir) : orig(origin), dir(dir) {}

	point3 origin() const { return orig; }
	vec3 direction() const { return dir; }

	point3 at(double t) const { return orig + t * dir; }

public:
	point3 orig;
	vec3 dir;
};
