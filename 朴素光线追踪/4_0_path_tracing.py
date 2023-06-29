import matplotlib
import cv2 as im
import taichi as ti
import numpy as np
import argparse
from ray_tracing_models import Ray, Camera, Hittable_list, Sphere, PI, random_in_unit_sphere, refract, reflect, reflectance, random_unit_vector
ti.init(arch=ti.gpu)

# Canvas
aspect_ratio = 1.0
image_width = 400
image_height = int(image_width / aspect_ratio)
canvas = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height))

# Rendering parameters
max_depth = 8
samples_per_pixel=8


@ti.kernel
def render():
    for i, j in canvas:
        color=get_color(i,j)
        canvas[i, j] += color

@ti.func
def get_color(i,j):
    color=ti.Vector([0.0,0.0,0.0])
    for n in range(samples_per_pixel):
        u = (i + ti.random()) / image_width
        v = (j + ti.random()) / image_height
        ray = camera.get_ray(u, v)
        color += ray_color(ray)
    color /= samples_per_pixel
    return color
# Path tracing
@ti.func
def ray_color(ray):
    color_buffer = ti.Vector([0.0, 0.0, 0.0])
    brightness = ti.Vector([1.0, 1.0, 1.0])
    scattered_origin = ray.origin
    scattered_direction = ray.direction
    p_RR = 0.9
    for n in range(max_depth):
        if ti.random() > p_RR:
            break
        is_hit, hit_point, hit_point_normal,\
            front_face, material, color = scene.hit(Ray(scattered_origin, scattered_direction))
        if is_hit:
            if material == 0:
                color_buffer = color * brightness
                break
            else:
                # Diffuse
                if material == 1:
                    target = hit_point + hit_point_normal
                    target += random_unit_vector()
                    scattered_direction = target - hit_point
                    scattered_origin = hit_point
                    brightness *= color
                # Metal and Fuzz Metal
                elif material == 2 or material == 4:
                    fuzz = 0.0
                    if material == 4:
                        fuzz = 0.4
                    scattered_direction = reflect(scattered_direction.normalized(),
                                                  hit_point_normal)
                    scattered_direction += fuzz * random_unit_vector()
                    scattered_origin = hit_point
                    if scattered_direction.dot(hit_point_normal) < 0:
                        break
                    else:
                        brightness *= color
                # Dielectric
                elif material == 3:
                    refraction_ratio = 1.5
                    if front_face:
                        refraction_ratio = 1 / refraction_ratio
                    cos_theta = min(-scattered_direction.normalized().dot(hit_point_normal), 1.0)
                    sin_theta = ti.sqrt(1 - cos_theta * cos_theta)
                    # total internal reflection
                    if refraction_ratio * sin_theta > 1.0 or reflectance(cos_theta, refraction_ratio) > ti.random():
                        scattered_direction = reflect(scattered_direction.normalized(), hit_point_normal)
                    else:
                        scattered_direction = refract(scattered_direction.normalized(), hit_point_normal, refraction_ratio)
                    scattered_origin = hit_point
                    brightness *= color
                brightness /= p_RR
    return color_buffer


if __name__ == "__main__":
    scene = Hittable_list()
    # Light source
    #scene.add(Sphere(center=ti.Vector([0, 5.4, -1]), radius=3.0, material=0, color=ti.Vector([10.0, 10.0, 10.0])))
    # Ground
    scene.add(Sphere(center=ti.Vector([0, -100.5, -1]), radius=100.0, material=1, color=ti.Vector([0.8, 0.8, 0.8])))
    # ceiling
    scene.add(Sphere(center=ti.Vector([0, 110.5, -1]), radius=100.0, material=1, color=ti.Vector([0.8, 0.8, 0.8])))
    # back wall
    scene.add(Sphere(center=ti.Vector([0, 1, 110]), radius=100.0, material=1, color=ti.Vector([0.8, 0.8, 0.8])))
    # right wall
    scene.add(Sphere(center=ti.Vector([-105.5, 0, -1]), radius=100.0, material=1, color=ti.Vector([0.6, 0.0, 0.0])))
    # left wall
    scene.add(Sphere(center=ti.Vector([105.5, 0, -1]), radius=100.0, material=1, color=ti.Vector([0.0, 0.6, 0.0])))

    # Metal ball
    scene.add(Sphere(center=ti.Vector([-0.8, 0.2, 2]), radius=0.7, material=2, color=ti.Vector([0.6, 0.8, 0.8])))
    # Glass ball
    scene.add(Sphere(center=ti.Vector([0.0, 0, -0.5]), radius=0.5, material=3, color=ti.Vector([1.0, 1.0, 1.0])))
    # light source
    scene.add(Sphere(center=ti.Vector([0.0, 0, -0.5]), radius=0.2, material=0, color=ti.Vector([2, 3, 5])))
    # Metal ball-2
    scene.add(Sphere(center=ti.Vector([1.0, -0.15, 1.6]), radius=0.4, material=4, color=ti.Vector([0.8, 0.6, 0.2])))
    # glass ball
    scene.add(Sphere(center=ti.Vector([0.8, 0.5, 3.0]), radius=0.8, material=3, color=ti.Vector([1.0, 1.0, 1.0])))
    # light source
    scene.add(Sphere(center=ti.Vector([0.8, 0.5, 3.0]), radius=0.4, material=0, color=ti.Vector([4, 8, 5])))
    # glass ball
    scene.add(Sphere(center=ti.Vector([1.0, 0.1, -2.0]), radius=0.6, material=3, color=ti.Vector([1.0, 1.0, 1.0])))
    # light source
    scene.add(Sphere(center=ti.Vector([1.0, 0.1, -2.0]), radius=0.3, material=0, color=ti.Vector([5, 3, 8])))
    # diffuse ball
    scene.add(Sphere(center=ti.Vector([-0.7, -0.1, -2.0]), radius=0.4, material=1, color=ti.Vector([0.4, 0.8, 0.6])))
    # diffuse ball
    scene.add(Sphere(center=ti.Vector([-1.5, -0.23, -0.5]), radius=0.3, material=1, color=ti.Vector([0.6, 0.4, 0.3])))
    # glass ball
    scene.add(Sphere(center=ti.Vector([1.9, -0.2, 0.8]), radius=0.4, material=3, color=ti.Vector([1.0, 1.0, 1.0])))
    # glass ball
    scene.add(Sphere(center=ti.Vector([-2.4, -0.0, 1.5]), radius=0.6, material=3, color=ti.Vector([1.0, 1.0, 1.0])))
    # light source
    scene.add(Sphere(center=ti.Vector([-2.4, -0.0, 1.5]), radius=0.3, material=0, color=ti.Vector([2, 3, 8])))
    camera = Camera()
    #gui = ti.GUI("Ray Tracing", res=(image_width, image_height))
    z=-7+3.5/300*27
    num=28
    while z<=-3/5:
        canvas.fill(0)
        img=np.sqrt(canvas.to_numpy())
        camera.reset(0.0,-6/12.25*(z+3.5)**2+6,z)
        for cnt in range(1,501):
            render()
            img = np.sqrt(canvas.to_numpy() / cnt)
            #gui.set_image(img)
            #gui.show()
        img1=np.zeros((image_height,image_width,3))
        for i in range(3):
            img1[:,:,i]=np.transpose(img[:,:,2-i])
            img1[:,:,i]=np.flipud(img1[:,:,i])*256
        im.imwrite("./img/output/%d.jpg"%num,img1)
        num=num+1
        z=z+3.5/300.0