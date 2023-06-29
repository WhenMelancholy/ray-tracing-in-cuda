#pragma once

#include <iostream>
#include "rtweekend.cuh"

#include <cuda.h>
#include <curand_kernel.h>

class vec3 {
public:
    // UPDATE 这个文件中的随机数相关代码均需要删除，使用 curand 库重新实现
    // 一个十分神奇的编译问题，假如删掉了下面这个空行，就会导致编译错误
    // 根据测试，可能是行末换行符+文件编码共同引起的问题
    static __device__ vec3 random(curandState *state) {
        return {random_float(state), random_float(state), random_float(state)};
    }

    static __device__ vec3 random(float min, float max, curandState *state) {
        return {random_float(min, max, state), random_float(min, max, state),
                random_float(min, max, state)};
    }

public:
    __host__ __device__ vec3() : e{0, 0, 0} {}

    __host__ __device__ vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}

    __host__ __device__ float x() const { return e[0]; }

    __host__ __device__ float y() const { return e[1]; }

    __host__ __device__ float z() const { return e[2]; }

    __host__ __device__ vec3 operator-() const {
        return vec3(-e[0], -e[1], -e[2]);
    }

    __host__ __device__ float operator[](int i) const { return e[i]; }

    __host__ __device__ float &operator[](int i) { return e[i]; }

    __host__ __device__ vec3 &operator+=(const vec3 &v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __host__ __device__ vec3 &operator*=(const float t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ vec3 &operator/=(const float t) {
        return *this *= 1 / t;
    }

    __host__ __device__ float length() const { return sqrt(length_squared()); }

    __host__ __device__ float length_squared() const {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    __host__ __device__ bool near_zero(const float &eps = 1e-8) {
        return fabs(e[0]) < eps && fabs(e[1]) < eps && fabs(e[2]) < eps;
    }

    __host__ __device__ vec3 normalize() const {
        float len = length();
        return vec3(e[0] / len, e[1] / len, e[2] / len);
    }

public:
    float e[3];
};

using point3 = vec3;
using color = vec3;

// begin utility functions
std::ostream &operator<<(std::ostream &out, const vec3 &v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ vec3 operator+(const vec3 &u, const vec3 &v) {
    return vec3(u.x() + v.x(), u.y() + v.y(), u.z() + v.z());
}

__host__ __device__ vec3 operator-(const vec3 &u, const vec3 &v) {
    return vec3(u.x() - v.x(), u.y() - v.y(), u.z() - v.z());
}

__host__ __device__ vec3 operator*(const vec3 &u, const vec3 &v) {
    return vec3(u.x() * v.x(), u.y() * v.y(), u.z() * v.z());
}

__host__ __device__ vec3 operator*(float t, const vec3 &v) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ vec3 operator*(const vec3 &v, float t) { return t * v; }

__host__ __device__ vec3 operator/(vec3 v, float t) { return (1 / t) * v; }

__host__ __device__ float dot(const vec3 &u, const vec3 &v) {
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

__host__ __device__ vec3 cross(const vec3 &u, const vec3 &v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ vec3 unit_vector(vec3 v) { return v / v.length(); }

__host__ __device__ vec3 reflect(const vec3 &v, const vec3 &n) {
    return v - 2 * dot(v, n) * n;
}

using std::min;

__host__ __device__ vec3 refract(const vec3 &uv, const vec3 &n,
                                 float etai_over_etat) {
    auto cos_theta = min(dot(-uv, n), 1.0f);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -sqrt(fabs(1.0f - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

__device__ vec3 random_in_unit_sphere(curandState *state) {
    for (;;) {
        auto p =
            vec3(random_float(state), random_float(state), random_float(state));
        if (p.length_squared() >= 1)
            continue;
        return p;
    }
}

__device__ vec3 random_unit_vector(curandState *state) {
    return unit_vector(random_in_unit_sphere(state));
}

__device__ vec3 random_in_unit_disk(curandState *state) {
    while (true) {
        auto p =
            vec3(random_float(-1, 1, state), random_float(-1, 1, state), 0);
        if (p.length_squared() >= 1)
            continue;
        return p;
    }
}

class ray;
struct matrix4x4 {
    __device__ __host__ matrix4x4() {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                m[i][j] = 0;
    }

    __device__ __host__ matrix4x4(float mat[4][4]) {
        memcpy(m, mat, 16 * sizeof(float));
    }

    __device__ __host__ matrix4x4(float t00, float t01, float t02, float t03,
                                  float t10, float t11, float t12, float t13,
                                  float t20, float t21, float t22, float t23,
                                  float t30, float t31, float t32, float t33) {
        m[0][0] = t00;
        m[0][1] = t01;
        m[0][2] = t02;
        m[0][3] = t03;
        m[1][0] = t10;
        m[1][1] = t11;
        m[1][2] = t12;
        m[1][3] = t13;
        m[2][0] = t20;
        m[2][1] = t21;
        m[2][2] = t22;
        m[2][3] = t23;
        m[3][0] = t30;
        m[3][1] = t31;
        m[3][2] = t32;
        m[3][3] = t33;
    }

    __device__ __host__ bool operator==(const matrix4x4 &mat) const {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                if (m[i][j] != mat.m[i][j])
                    return false;
        return true;
    }

    __device__ __host__ bool operator!=(const matrix4x4 &mat) const {
        return !(*this == mat);
    }

    // Transpose this matrix
    __device__ __host__ matrix4x4 transpose() const {
        return matrix4x4(m[0][0], m[1][0], m[2][0], m[3][0], m[0][1], m[1][1],
                         m[2][1], m[3][1], m[0][2], m[1][2], m[2][2], m[3][2],
                         m[0][3], m[1][3], m[2][3], m[3][3]);
    }

    // Multiply two matrices
    __device__ __host__ matrix4x4 operator*(const matrix4x4 &rhs) const {
        matrix4x4 r;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                r.m[i][j] = m[i][0] * rhs.m[0][j] + m[i][1] * rhs.m[1][j] +
                            m[i][2] * rhs.m[2][j] + m[i][3] * rhs.m[3][j];
        return r;
    }

    // Get inverse of this matrix
    __device__ __host__ matrix4x4 inverse() const {
        int indxc[4], indxr[4];
        int ipiv[4] = {0, 0, 0, 0};
        float minv[4][4];
        memcpy(minv, m, 4 * 4 * sizeof(float));
        for (int i = 0; i < 4; i++) {
            int irow = 0, icol = 0;
            float big = 0.f;
            // Choose pivot
            for (int j = 0; j < 4; j++) {
                if (ipiv[j] != 1) {
                    for (int k = 0; k < 4; k++) {
                        if (ipiv[k] == 0) {
                            if (abs(minv[j][k]) >= big) {
                                big = float(abs(minv[j][k]));
                                irow = j;
                                icol = k;
                            }
                        } else if (ipiv[k] > 1)
                            printf("Singular matrix in MatrixInvert");
                    }
                }
            }
            ++ipiv[icol];
            // Swap rows _irow_ and _icol_ for pivot
            if (irow != icol) {
                for (int k = 0; k < 4; ++k) {
                    auto tmp = minv[irow][k];
                    minv[irow][k] = minv[icol][k];
                    minv[icol][k] = tmp;
                }
            }
            indxr[i] = irow;
            indxc[i] = icol;
            if (minv[icol][icol] == 0.f)
                printf("Singular matrix in MatrixInvert");

            // Set $m[icol][icol]$ to one by scaling row _icol_ appropriately
            float pivinv = 1. / minv[icol][icol];
            minv[icol][icol] = 1.;
            for (int j = 0; j < 4; j++)
                minv[icol][j] *= pivinv;

            // Subtract this row from others to zero out their columns
            for (int j = 0; j < 4; j++) {
                if (j != icol) {
                    float save = minv[j][icol];
                    minv[j][icol] = 0;
                    for (int k = 0; k < 4; k++)
                        minv[j][k] -= minv[icol][k] * save;
                }
            }
        }
        // Swap columns to reflect permutation
        for (int j = 3; j >= 0; j--) {
            if (indxr[j] != indxc[j]) {
                for (int k = 0; k < 4; k++) {
                    auto tmp = minv[k][indxr[j]];
                    minv[k][indxr[j]] = minv[k][indxc[j]];
                    minv[k][indxc[j]] = tmp;
                }
            }
        }
        return matrix4x4(minv);
    }

    // Check if matrix is identity
    __device__ __host__ bool is_identity() const {
        return m[0][0] == 1.f && m[0][1] == 0.f && m[0][2] == 0.f &&
               m[0][3] == 0.f && m[1][0] == 0.f && m[1][1] == 1.f &&
               m[1][2] == 0.f && m[1][3] == 0.f && m[2][0] == 0.f &&
               m[2][1] == 0.f && m[2][2] == 1.f && m[2][3] == 0.f &&
               m[3][0] == 0.f && m[3][1] == 0.f && m[3][2] == 0.f &&
               m[3][3] == 1.f;
    }

    float m[4][4];
};

class transform {
public:
    __device__ __host__ transform() {}

    __device__ __host__ transform(const float mat[4][4]) {
        m = matrix4x4(mat[0][0], mat[0][1], mat[0][2], mat[0][3], mat[1][0],
                      mat[1][1], mat[1][2], mat[1][3], mat[2][0], mat[2][1],
                      mat[2][2], mat[2][3], mat[3][0], mat[3][1], mat[3][2],
                      mat[3][3]);
        m_inv = m.inverse();
    }

    __device__ __host__ transform(const matrix4x4 &mat)
        : m(mat), m_inv(mat.inverse()) {}

    __device__ __host__ transform(const matrix4x4 &mat, const matrix4x4 &minv)
        : m(mat), m_inv(minv) {}

    __device__ __host__ transform inverse() const {
        return transform(m_inv, m);
    }

    __device__ __host__ transform transpose() const {
        return transform(m.transpose(), m_inv.transpose());
    }

    __device__ __host__ bool is_identity() const { return m.is_identity(); }

    // operator == and operator !=
    __device__ __host__ bool operator==(const transform &t) const {
        return t.m == m && t.m_inv == m_inv;
    }

    __device__ __host__ bool operator!=(const transform &t) const {
        return t.m != m || t.m_inv != m_inv;
    }

    // get matrix or get inverse matrix
    __device__ __host__ const matrix4x4 &get_matrix() const { return m; }

    __device__ __host__ const matrix4x4 &get_inverse_matrix() const {
        return m_inv;
    }

    // operator *
    __device__ __host__ transform operator*(const transform &t) const {
        return transform(m * t.m, t.m_inv * m_inv);
    }

    // operator () to transform point, vector, normal, ray, bound3
    __device__ __host__ point3 operator()(const point3 &p) const {
        float x = p.x(), y = p.y(), z = p.z();
        float xp = m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z + m.m[0][3];
        float yp = m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z + m.m[1][3];
        float zp = m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z + m.m[2][3];
        float wp = m.m[3][0] * x + m.m[3][1] * y + m.m[3][2] * z + m.m[3][3];
        if (wp == 1.)
            return point3(xp, yp, zp);
        else
            return point3(xp, yp, zp) / wp;
    }

    __device__ __host__ vec3 apply_point(const point3 &p) const {
        return (*this)(p);
    }

    __device__ __host__ vec3 apply_vec(const vec3 &p) const {
        float x = p.x(), y = p.y(), z = p.z();
        return vec3(m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z,
                    m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z,
                    m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z);
    }

    __device__ __host__ ray apply_ray(const ray &r) const;

    // apply transform to normal vector p
    __device__ __host__ vec3 apply_normal(const vec3 &p) const {
        float x = p.x(), y = p.y(), z = p.z();
        return vec3(m_inv.m[0][0] * x + m_inv.m[1][0] * y + m_inv.m[2][0] * z,
                    m_inv.m[0][1] * x + m_inv.m[1][1] * y + m_inv.m[2][1] * z,
                    m_inv.m[0][2] * x + m_inv.m[1][2] * y + m_inv.m[2][2] * z);
    }

private:
    matrix4x4 m, m_inv;
};

// common transforms
__device__ __host__ transform translate(const vec3 &delta) {
    matrix4x4 m(1, 0, 0, delta.x(), 0, 1, 0, delta.y(), 0, 0, 1, delta.z(), 0,
                0, 0, 1);
    matrix4x4 minv(1, 0, 0, -delta.x(), 0, 1, 0, -delta.y(), 0, 0, 1,
                   -delta.z(), 0, 0, 0, 1);
    return transform(m, minv);
}

__device__ __host__ transform rotate(const vec3 &axis, float theta) {
    vec3 a = axis.normalize();
    float sintheta = sin(theta);
    float costheta = cos(theta);
    matrix4x4 m;
    m.m[0][0] = a.x() * a.x() + (1 - a.x() * a.x()) * costheta;
    m.m[0][1] = a.x() * a.y() * (1 - costheta) - a.z() * sintheta;
    m.m[0][2] = a.x() * a.z() * (1 - costheta) + a.y() * sintheta;
    m.m[0][3] = 0;
    m.m[1][0] = a.x() * a.y() * (1 - costheta) + a.z() * sintheta;
    m.m[1][1] = a.y() * a.y() + (1 - a.y() * a.y()) * costheta;
    m.m[1][2] = a.y() * a.z() * (1 - costheta) - a.x() * sintheta;
    m.m[1][3] = 0;
    m.m[2][0] = a.x() * a.z() * (1 - costheta) - a.y() * sintheta;
    m.m[2][1] = a.y() * a.z() * (1 - costheta) + a.x() * sintheta;
    m.m[2][2] = a.z() * a.z() + (1 - a.z() * a.z()) * costheta;
    m.m[2][3] = 0;
    m.m[3][0] = 0;
    m.m[3][1] = 0;
    m.m[3][2] = 0;
    m.m[3][3] = 1;
    return transform(m, m.transpose());
}

__device__ __host__ transform scale(float x, float y, float z) {
    matrix4x4 m(x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1);
    matrix4x4 minv(1.f / x, 0, 0, 0, 0, 1.f / y, 0, 0, 0, 0, 1.f / z, 0, 0, 0,
                   0, 1);
    return transform(m, minv);
}

__device__ __host__ transform identity() { return translate({0, 0, 0}); }