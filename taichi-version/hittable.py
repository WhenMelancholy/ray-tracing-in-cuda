import taichi as ti
from vector import *
import ray
from material import Materials
import random
import numpy as np
from bvh import BVH
import cv2

@ti.func
def is_front_facing(ray_direction, normal):
    return ray_direction.dot(normal) < 0.0


@ti.func
def hit_sphere(center, radius, ray_origin, ray_direction, t_min, t_max):
    ''' Intersect a sphere of given radius and center and return
        if it hit and the least root. '''
    oc = ray_origin - center
    a = ray_direction.norm_sqr()
    half_b = oc.dot(ray_direction)
    c = (oc.norm_sqr() - radius**2)
    discriminant = (half_b**2) - a * c

    hit = discriminant >= 0.0
    root = -1.0
    if hit:
        sqrtd = discriminant**0.5
        root = (-half_b - sqrtd) / a

        if root < t_min or t_max < root:
            root = (-half_b + sqrtd) / a
            if root < t_min or t_max < root:
                hit = False
    return hit, root


@ti.func
def hit_triangle(v1,v2,v3,nn, ray_origin, ray_direction, t_min, t_max):
    ''' Intersect a triangle of given v and n and return
        if it hit and the least root. '''
    n = nn
    hit = False
    root = -1.0
    w = Vector(0.33,0.33,0.34)
    oc = ray_origin - v1
    if oc.dot(n)<0:
        n = -n
    a = ray_direction.norm()   
    theta =  ray_direction.dot(n)/a  
    if theta<0:
        r_interact = ray_origin-ray_direction/a*oc.dot(n)/theta
        n1 =(r_interact-v1).cross(v2-v1).dot((v3-v1).cross(v2-v1))
        n2 = (r_interact-v2).cross(v1-v2).dot((v3-v2).cross(v1-v2))
        n3 =(r_interact-v1).cross(v3-v1).dot((v2-v1).cross(v3-v1))
        n4 = (r_interact-v2).cross(v3-v2).dot((v1-v2).cross(v3-v2))
        w1 = (r_interact-v1).cross(r_interact-v2).norm()/(v3-v1).cross(v3-v2).norm()
        w2 = (r_interact-v1).cross(r_interact-v3).norm()/(v2-v1).cross(v2-v3).norm()
        w3 = (r_interact-v3).cross(r_interact-v2).norm()/(v1-v3).cross(v1-v2).norm()
        w = Vector(w1,w2,w3)
        if n1 > 0 and n2 > 0 and n3 > 0 and n4 > 0:
            hit = True
            root = -oc.dot(n)/theta/a
        else:
            hit = False
            root = -1
        if root < t_min or root > t_max:
            hit = False
            root = -1
    
    return hit, root , w

class Sphere:
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material
        self.id = -1
        self.box_min = [
            self.center[0] - radius, self.center[1] - radius,
            self.center[2] - radius
        ]
        self.box_max = [
            self.center[0] + radius, self.center[1] + radius,
            self.center[2] + radius
        ]

    @property
    def bounding_box(self):
        return self.box_min, self.box_max

class Triangle:
    def __init__(self, v1, v2, v3 ,u1, u2, u3 ,material):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.u1 = u1
        self.u2 = u2
        self.u3 = u3
        self.radius = 0
        self.normal =(v2-v1).cross(v3-v1)/(v2-v1).cross(v3-v1).norm()
        self.material = material
        self.id = -1
        self.box_min = [
            min(v1(0),v2(0),v3(0)),min(v1(1),v2(1),v3(1)),min(v1(2),v2(2),v3(2))
        ]
        self.box_max = [
            max(v1(0),v2(0),v3(0)),max(v1(1),v2(1),v3(1)),max(v1(2),v2(2),v3(2))
        ]
        self.center = Point(self.box_max+self.box_min)/2

    @property
    def bounding_box(self):
        return self.box_min, self.box_max


BRANCH = 1.0
LEAF = 0.0


@ti.data_oriented
class World:
    def __init__(self):
        self.spheres = []

    def clear(self):
        self.spheres = []

    def add(self, sphere):
        sphere.id = len(self.spheres)
        self.spheres.append(sphere)

    def commit(self):
        ''' Commit should be called after all objects added.  
            Will compile bvh and materials. '''
        self.n = len(self.spheres)

        self.materials = Materials(self.n)
        self.bvh = BVH(self.spheres)
        self.radius = ti.field(ti.f32)
        self.center = ti.Vector.field(3, dtype=ti.f32)
        self.v1 = ti.Vector.field(3, dtype=ti.f32)
        self.v2 = ti.Vector.field(3, dtype=ti.f32)
        self.v3 = ti.Vector.field(3, dtype=ti.f32)
        self.u1 = ti.Vector.field(2, dtype=ti.f32)
        self.u2 = ti.Vector.field(2, dtype=ti.f32)
        self.u3 = ti.Vector.field(2, dtype=ti.f32)
        self.nor= ti.Vector.field(3, dtype=ti.f32)
        ti.root.dense(ti.i, self.n).place(self.radius, self.center,self.v1,self.v2,self.v3,self.u1,self.u2,self.u3,self.nor)

        self.bvh.build()

        for i in range(self.n):
            self.center[i] = self.spheres[i].center
            self.radius[i] = self.spheres[i].radius
            if self.radius[i]==0:
                self.v1[i] = self.spheres[i].v1
                self.v2[i] = self.spheres[i].v2
                self.v3[i] = self.spheres[i].v3
                self.u1[i] = self.spheres[i].u1
                self.u2[i] = self.spheres[i].u2
                self.u3[i] = self.spheres[i].u3
                self.nor[i] = self.spheres[i].normal
            self.materials.set(i, self.spheres[i].material)
        texture = cv2.imread('.\\asset\\tex\\bricks2.png',1)
        # print(texture)
        # cv2.imshow('imshow',texture)
        # cv2.waitKey(0)
        for i in range(100):
            for j in range(100):
                #print(texture[i][j])
                self.materials.settexture(0,i,j, texture[i][j])
        # self.materials.showtexture(0)
        del self.spheres

    def update(self):
        self.n = len(self.spheres)

        self.bvh = BVH(self.spheres)
        self.bvh.build()
        for i in range(self.n):
            self.center[i] = self.spheres[i].center
            self.radius[i] = self.spheres[i].radius
            if self.radius[i]==0:
                self.v1[i] = self.spheres[i].v1
                self.v2[i] = self.spheres[i].v2
                self.v3[i] = self.spheres[i].v3
                self.u1[i] = self.spheres[i].u1
                self.u2[i] = self.spheres[i].u2
                self.u3[i] = self.spheres[i].u3
                self.nor[i] = self.spheres[i].normal
            self.materials.set(i, self.spheres[i].material)
        del self.spheres

    def bounding_box(self, i):
        return self.bvh_min(i), self.bvh_max(i)

    @ti.func
    def hit_all(self, ray_origin, ray_direction):
        ''' Intersects a ray against all objects. '''
        hit_anything = False
        t_min = 0.0001
        closest_so_far = 9999999999.9
        hit_index = 0
        p = Point(0.0, 0.0, 0.0)
        n = Vector(0.0, 0.0, 0.0)
        front_facing = True
        i = 0
        curr = self.bvh.bvh_root
        w = Vector(0.0,0.0,0.0)
        uv = Vector2(0.0,0.0)
        # walk the bvh tree
        while curr != -1:
            obj_id, left_id, right_id, next_id = self.bvh.get_full_id(curr)
            #print(obj_id)
            
            #print(ti.cast(obj_id,ti.i32))
            if obj_id != -1:
                # this is a leaf node, check the sphere
                hit = True
                t = 0.0
                
                if self.radius[obj_id] != 0:
                    hit, t = hit_sphere(self.center[obj_id], self.radius[obj_id],
                                        ray_origin, ray_direction, t_min,
                                        closest_so_far)                   
                else:
                    hit, t ,w = hit_triangle(self.v1[obj_id],self.v2[obj_id],self.v3[obj_id],self.nor[obj_id],
                                        ray_origin, ray_direction, t_min,
                                        closest_so_far)
                if hit:
                    
                    uv = self.u1[obj_id]*w[0] + self.u2[obj_id]*w[1] + self.u3[obj_id]*w[2]
                    
                    hit_anything = True
                    closest_so_far = t
                    hit_index = obj_id
                curr = next_id
            else:
                if self.bvh.hit_aabb(curr, ray_origin, ray_direction, t_min,
                                     closest_so_far):
                    # add left and right children
                    if left_id != -1:
                        curr = left_id
                    elif right_id != -1:
                        curr = right_id
                    else:
                        curr = next_id
                else:
                    curr = next_id

        if hit_anything:
            if self.radius[hit_index] != 0:
                p = ray.at(ray_origin, ray_direction, closest_so_far)
                n = (p - self.center[hit_index]) / self.radius[hit_index]
                front_facing = is_front_facing(ray_direction, n)
                n = n if front_facing else -n
            else:
                p = ray.at(ray_origin, ray_direction, closest_so_far)
                n = self.nor[hit_index]
                front_facing = is_front_facing(ray_direction, n)
                n = n if front_facing else -n
        
        return hit_anything, p, uv,n, front_facing, hit_index

    @ti.func
    def scatter(self, ray_direction, p, uv,n, front_facing, index):
        ''' Get the scattered direction for a ray hitting an object '''
        return self.materials.scatter(index, ray_direction, p, uv,n, front_facing)
