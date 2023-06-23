#pragma once

#include "material.cuh"
#include "vec3.cuh"
#include "rtweekend.cuh"
#include "texture.cuh"
#include "object.cuh"
#include "color.cuh"
#include "camera.cuh"

#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct scene {
    int num_of_objects;
    int num_of_materials;
    int num_of_textures;
    hittable **objects;
    material **materials;
    mytexture **textures;

    hittable_list **world;
    camera **cam;
    int max_depth;

    int samples_per_pixel;
    int width;
    int height;
    color background;
};

scene *parse_file(const std::string &filename) {
    std::ifstream file(filename);
    json data = json::parse(file);

    scene host_scene;

    host_scene.background = color(data["background"][0], data["background"][1], data["background"][2]);
    host_scene.max_depth = data["max_depth"];
    host_scene.samples_per_pixel = data["samples_per_pixel"];
    host_scene.width = data["width"];
    host_scene.height = data["height"];

    host_scene.num_of_objects = data["object"].size();
    host_scene.num_of_materials = data["material"].size();
    host_scene.num_of_textures = data["texture"].size();

    when("num_of_objects: %d\n", host_scene.num_of_objects);
    when("num_of_materials: %d\n", host_scene.num_of_materials);
    when("num_of_textures: %d\n", host_scene.num_of_textures);

    // create camera on cpu
    host_scene.cam = new camera *[1];
    auto camdata = data["camera"];
    auto lookfrom = point3(camdata["lookfrom"][0], camdata["lookfrom"][1], camdata["lookfrom"][2]);
    auto lookat = point3(camdata["lookat"][0], camdata["lookat"][1], camdata["lookat"][2]);
    auto vup = vec3(camdata["vup"][0], camdata["vup"][1], camdata["vup"][2]);
    auto vfov = camdata["vfov"];
    auto aspect_ratio = float(host_scene.width) / float(host_scene.height);
    auto aperture = camdata["aperture"];
    auto focus_dist = (lookfrom - lookat).length();
    host_scene.cam[0] = new camera(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, focus_dist);

    // create textures on cpu

    // create materials on cpu
    host_scene.materials = new material *[host_scene.num_of_materials];
    for (int i = 0; i < host_scene.num_of_materials; ++i) {
        auto matdata = data["material"][i];
        if (matdata["type"] == "lambertian") {
            auto albedo = color(matdata["albedo"][0].get<double>(),
                                matdata["albedo"][1].get<double>(),
                                matdata["albedo"][2].get<double>());
            when("creating lambertian, albedo: %f %f %f\n", albedo.x(), albedo.y(), albedo.z());
            host_scene.materials[i] = new lambertian(albedo);
        }
        if (matdata["type"].get<std::string>() == "metal") {
            auto albedo = color(matdata["albedo"][0].get<double>(),
                                matdata["albedo"][1].get<double>(),
                                matdata["albedo"][2].get<double>());
            auto fuzz = matdata["fuzz"].get<double>();
            when("creating metal, albedo: %f %f %f, fuzz: %f\n", albedo.x(), albedo.y(), albedo.z(),
                 fuzz);
            host_scene.materials[i] = new metal(albedo, fuzz);
        }
        if (matdata["type"].get<std::string>() == "dielectric") {
            auto ir = matdata["index_of_refraction"].get<double>();
            when("creating dielectric, ir: %f\n", ir);
            host_scene.materials[i] = new dielectric(ir);
        }
    }

    // create objects on cpu
    host_scene.objects = new hittable *[host_scene.num_of_objects];
    for (int i = 0; i < host_scene.num_of_objects; ++i) {
        auto objdata = data["object"][i];
        if (objdata["type"].get<std::string>() == "sphere") {
            auto center = point3(objdata["center"][0].get<double>(),
                                 objdata["center"][1].get<double>(),
                                 objdata["center"][2].get<double>());
            auto radius = objdata["radius"].get<double>();
            auto material_id = objdata["material"].get<int>();
            when("creating sphere, center: %f %f %f, radius: %f, material: %d\n", center.x(), center.y(), center.z(),
                 radius, material_id);
            host_scene.objects[i] = new sphere(center, radius, host_scene.materials[material_id]);
        }
    }

    // deep copy scene to gpu
    scene *dev_scene;
    checkCudaErrors(cudaMalloc((void **) &dev_scene, sizeof(scene)));
    checkCudaErrors(cudaMemcpy(dev_scene, &host_scene, sizeof(scene), cudaMemcpyHostToDevice));

    // copy camera to gpu
    camera **dev_cam;
    dev_cam = new camera *[1];
    checkCudaErrors(cudaMalloc((void **) &dev_cam[0], sizeof(camera)));
    checkCudaErrors(cudaMemcpy(dev_cam[0], host_scene.cam[0], sizeof(camera), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(&dev_scene->cam, &dev_cam, sizeof(camera *), cudaMemcpyHostToDevice));

    // copy materials to gpu
    material **dev_materials;
    dev_materials = new material *[host_scene.num_of_materials];
    for (int i = 0; i < host_scene.num_of_materials; ++i) {
        checkCudaErrors(cudaMalloc((void **) &dev_materials[i], sizeof(material)));
        checkCudaErrors(
                cudaMemcpy(dev_materials[i], host_scene.materials[i], sizeof(material), cudaMemcpyHostToDevice));
    }
    checkCudaErrors(cudaMemcpy(&dev_scene->materials, &dev_materials, sizeof(material *), cudaMemcpyHostToDevice));

    // copy objects to gpu
    hittable **dev_objects;
    dev_objects = new hittable *[host_scene.num_of_objects];
    for (int i = 0; i < host_scene.num_of_objects; ++i) {
        if (data["object"][i]["type"].get<std::string>() == "sphere") {
            sphere *obj = (sphere *) host_scene.objects[i];
            obj->mat_ptr = dev_materials[i];
        }
        checkCudaErrors(cudaMalloc((void **) &dev_objects[i], sizeof(hittable)));
        checkCudaErrors(cudaMemcpy(dev_objects[i], host_scene.objects[i], sizeof(hittable), cudaMemcpyHostToDevice));
    }
    checkCudaErrors(cudaMemcpy(&dev_scene->objects, &dev_objects, sizeof(hittable *), cudaMemcpyHostToDevice));

    return nullptr;
}