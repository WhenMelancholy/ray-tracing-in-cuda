#include "vec3.cuh"
#include <cuda.h>

class aabb {
public:
    __device__ __host__ aabb() {}
    __device__ __host__ aabb(const point3 &a, const point3 &b) {
        minimum = a;
        maximum = b;
    }

    __device__ __host__ point3 min() const { return minimum; }
    __device__ __host__ point3 max() const { return maximum; }

    __device__ __host__ bool hit(const ray &r, double t_min,
                                 double t_max) const {
        for (int a = 0; a < 3; a++) {
            auto invD = 1.0f / r.direction()[a];
            auto t0 = (min()[a] - r.origin()[a]) * invD;
            auto t1 = (max()[a] - r.origin()[a]) * invD;
            if (invD < 0.0f) {
                auto temp = t0;
                t0 = t1;
                t1 = temp;
            }
            t_min = t0 > t_min ? t0 : t_min;
            t_max = t1 < t_max ? t1 : t_max;
            if (t_max <= t_min)
                return false;
        }
        return true;
    }

    point3 minimum;
    point3 maximum;
};

__device__ aabb surrounding_box(aabb box0, aabb box1) {
    point3 small(fmin(box0.min().x(), box1.min().x()),
                 fmin(box0.min().y(), box1.min().y()),
                 fmin(box0.min().z(), box1.min().z()));

    point3 big(fmax(box0.max().x(), box1.max().x()),
               fmax(box0.max().y(), box1.max().y()),
               fmax(box0.max().z(), box1.max().z()));

    return aabb(small, big);
}