import taichi as ti
from taichi_glsl.vector import reflect
from vector import *
import cv2

@ti.func
def reflectance(cosine, idx):
    r0 = ((1.0 - idx) / (1.0 + idx))**2
    return r0 + (1.0 - r0) * ((1.0 - cosine)**5)


@ti.func
def reflect(v, n):
    return v - 2.0 * v.dot(n) * n


@ti.func
def refract(v, n, etai_over_etat):
    cos_theta = min(-v.dot(n), 1.0)
    r_out_perp = etai_over_etat * (v + cos_theta * n)
    r_out_parallel = -ti.sqrt(abs(1.0 - r_out_perp.norm_sqr())) * n
    return r_out_perp + r_out_parallel


class _material:
    def scatter(self, in_direction, p, n):
        pass


class Lambert(_material):
    def __init__(self, color,index = 0):
        self.color = color
        self.index = index
        self.roughness = 0.0
        self.ior = 1.0

    @staticmethod
    @ti.func
    def scatter(in_direction, p, n, color):
        out_direction = n + random_in_hemisphere(n)
        attenuation = color
        return True, p, out_direction, attenuation


class Metal(_material):
    def __init__(self, color, roughness):
        self.color = color
        self.index = 1
        self.roughness = min(roughness, 1.0)
        self.ior = 1.0

    @staticmethod
    @ti.func
    def scatter(in_direction, p, n, color, roughness):
        out_direction = reflect(in_direction.normalized(),
                                n) + roughness * random_in_unit_sphere()
        attenuation = color
        reflected = out_direction.dot(n) > 0.0
        return reflected, p, out_direction, attenuation


class Dielectric(_material):
    def __init__(self, ior):
        self.color = Color(1.0, 1.0, 1.0)
        self.index = 2
        self.roughness = 0.0
        self.ior = ior

    @staticmethod
    @ti.func
    def scatter(in_direction, p, n, color, ior, front_facing):
        refraction_ratio = 1.0 / ior if front_facing else ior
        unit_dir = in_direction.normalized()
        cos_theta = min(-unit_dir.dot(n), 1.0)
        sin_theta = ti.sqrt(1.0 - cos_theta * cos_theta)

        out_direction = Vector(0.0, 0.0, 0.0)
        cannot_refract = refraction_ratio * sin_theta > 1.0
        if cannot_refract or reflectance(cos_theta,
                                         refraction_ratio) > ti.random():
            out_direction = reflect(unit_dir, n)
        else:
            out_direction = refract(unit_dir, n, refraction_ratio)
        attenuation = color

        return True, p, out_direction, attenuation


@ti.data_oriented
class Materials:
    ''' List of materials for a scene.'''
    def __init__(self, n):
        self.roughness = ti.field(ti.f32)
        self.colors = ti.Vector.field(3, dtype=ti.f32)
        self.mat_index = ti.field(ti.u32)
        self.ior = ti.field(ti.f32)
        self.texture = ti.Vector.field(3,dtype=ti.uint8)
        ti.root.dense(ti.i, n).place(self.roughness, self.colors,
                                     self.mat_index, self.ior)
        ti.root.dense(ti.ijk, (1,100,100)).place(self.texture)
        #ti.root.dense(ti.i,1).dense(ti.j,400).dense(ti.k,400).place(self.texture)

    def set(self, i, material):
        self.colors[i] = material.color
        self.mat_index[i] = material.index
        self.roughness[i] = material.roughness
        self.ior[i] = material.ior

    def settexture(self, i , j , k , texture):
        self.texture[i,j,k] = Vector(texture[0],texture[1],texture[2])

    def showtexture(self, i):
        for j in range(100):
            for k in range(100):
                print(self.texture[i,j,k])
    @ti.func
    def scatter(self, i, ray_direction, p, uv,n, front_facing):
        ''' Get the scattered ray that hits a material '''
        mat_index = self.mat_index[i]
        color = self.colors[i]
        roughness = self.roughness[i]
        ior = self.ior[i]
        reflected = True
        out_origin = Point(0.0, 0.0, 0.0)
        out_direction = Vector(0.0, 0.0, 0.0)
        attenuation = Color(0.0, 0.0, 0.0)

        if mat_index == 0:
            reflected, out_origin, out_direction, attenuation = Lambert.scatter(
                ray_direction, p, n, color)
        elif mat_index == 1:
            reflected, out_origin, out_direction, attenuation = Metal.scatter(
                ray_direction, p, n, color, roughness)
        elif mat_index == 2:
            reflected, out_origin, out_direction, attenuation = Dielectric.scatter(
                ray_direction, p, n, color, ior, front_facing)
        else :
            x = int((uv[0]-ti.floor(uv[0]))*100)
            y = int((uv[1]-ti.floor(uv[1]))*100)
            
            color = Color((self.texture[0,x,y][2])/255.0,(self.texture[0,x,y][1])/255.0,(self.texture[0,x,y][0])/255.0)
            #print(color)
            reflected, out_origin, out_direction, attenuation = Lambert.scatter(
                ray_direction, p, n, color)
        return reflected, out_origin, out_direction, attenuation
