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

/**
 * sample json data
{
    "background": [0.1, 0.1, 0.1],
    "max_depth": 50,
    "samples_per_pixel": 100,
    "width": 1600,
    "height": 900,
    "camera": {
        "lookfrom": [2, 2, -13],
        "lookat": [0, 0, 0],
        "vup": [0, 1, 0],
        "vfov": 20,
        "aperture": 0.1
    },
    "object": [
        {
            "type": "sphere",
            "center": [0, 0, -1],
            "radius": 0.5,
            "material": 0
        },
        {
            "type": "sphere",
            "center": [0, -100.5, -1],
            "radius": 100,
            "material": 1
        },
        {
            "type": "sphere",
            "center": [1, 0, -1],
            "radius": 0.5,
            "material": 2
        },
        {
            "type": "sphere",
            "center": [-1, 0, -1],
            "radius": 0.5,
            "material": 3
        },
        {
            "type": "sphere",
            "center": [-1, 0, -1],
            "radius": -0.45,
            "material": 3
        }
    ],
    "material": [
        {
            "type": "lambertian",
            "texture": 0
        },
        {
            "type": "lambertian",
            "texture": 1
        }, {
            "type": "metal",
            "albedo": [0.8, 0.6, 0.2],
            "fuzz": 0.0
        }, {
            "type": "dielectric",
            "index_of_refraction": 1.5
        }
    ],
    "texture": [
        {
            "type": "solid_color",
            "color_value": [0.1, 0.2, 0.5]
        },
        {
            "type": "solid_color",
            "color_value": [0.8, 0.8, 0.0]
        }, {
            "type": "solid_color",
            "color_value": [0.8, 0.6, 0.2]
        }
    ]
}
 */
camera **parser_camera(json &data) {
    camera **host_cam = new camera *[1];
    auto camdata = data["camera"];
    auto lookfrom = point3(camdata["lookfrom"][0], camdata["lookfrom"][1], camdata["lookfrom"][2]);
    auto lookat = point3(camdata["lookat"][0], camdata["lookat"][1], camdata["lookat"][2]);
    auto vup = vec3(camdata["vup"][0], camdata["vup"][1], camdata["vup"][2]);
    auto vfov = camdata["vfov"];
    auto aspect_ratio = float(data["width"]) / float(data["height"]);
    auto aperture = camdata["aperture"];
    auto focus_dist = (lookfrom - lookat).length();
    host_cam[0] = new camera(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, focus_dist);

    camera **dev_copy, **dev_cam;
    dev_copy = new camera *[1];
    checkCudaErrors(cudaMalloc((void **) &dev_copy[0], sizeof(camera)));
    checkCudaErrors(cudaMemcpy(dev_copy[0], host_cam[0], sizeof(camera), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &dev_cam, sizeof(camera *)));
    checkCudaErrors(cudaMemcpy(dev_cam, &dev_copy, sizeof(camera *), cudaMemcpyHostToDevice));

    data["camera"]["host_ptr"] = reinterpret_cast<uintptr_t>(host_cam);
    data["camera"]["device_ptr"] = reinterpret_cast<uintptr_t>(dev_cam);

    return dev_cam;
}

mytexture **parser_texture(json &data) {
    mytexture **host_textures = new mytexture *[data["num_of_textures"]];
    for (int i = 0; i < data["num_of_textures"]; ++i) {
        auto texdata = data["texture"][i];
        if (texdata["type"] == "solid_color") {
            auto albedo = color(texdata["color"][0].get<double>(),
                                texdata["color"][1].get<double>(),
                                texdata["color"][2].get<double>());
            host_textures[i] = new solid_color(albedo);
            data["texture"][i]["host_ptr"] = reinterpret_cast<uintptr_t>(host_textures[i]);

            mytexture *dev_texture;
            checkCudaErrors(cudaMalloc((void **) &dev_texture, sizeof(solid_color)));
            checkCudaErrors(cudaMemcpy(dev_texture, host_textures[i], sizeof(solid_color), cudaMemcpyHostToDevice));
            data["texture"][i]["device_ptr"] = reinterpret_cast<uintptr_t>(dev_texture);
        }
    }

    mytexture **dev_copy, **dev_textures;
    dev_copy = new mytexture *[data["num_of_textures"]];
    for (int i = 0; i < data["num_of_textures"].get<int>(); ++i) {
        dev_copy[i] = reinterpret_cast<mytexture *>(data["texture"][i]["device_ptr"].get<uintptr_t>());
    }
    checkCudaErrors(cudaMalloc((void **) &dev_textures, sizeof(mytexture *) * data["num_of_textures"].get<int>()));
    checkCudaErrors(cudaMemcpy(dev_textures, dev_copy, sizeof(mytexture *) * data["num_of_textures"].get<int>(),
                               cudaMemcpyHostToDevice));

    data["textures"]["host_ptr"] = reinterpret_cast<uintptr_t>(host_textures);
    data["textures"]["device_ptr"] = reinterpret_cast<uintptr_t>(dev_textures);
    return host_textures;
}

material **parser_material(json &data) {
    material **host_materials = new material *[data["num_of_materials"]];
    for (int i = 0; i < data["num_of_materials"]; ++i) {
        auto matdata = data["material"][i];
        if (matdata["type"] == "lambertian") {
            auto texture_id = matdata["texture"].get<int>();
            host_materials[i] = new lambertian(
                    reinterpret_cast<mytexture *>(data["textures"][texture_id]["host_ptr"].get<uintptr_t>()));
            data["material"][i]["host_ptr"] = reinterpret_cast<uintptr_t>(host_materials[i]);

            material *dev_copy, *dev_material;
            dev_copy = new lambertian(
                    reinterpret_cast<mytexture *>(data["textures"][texture_id]["device_ptr"].get<uintptr_t>()));
            checkCudaErrors(cudaMalloc((void **) &dev_material, sizeof(lambertian)));
            checkCudaErrors(cudaMemcpy(dev_material, dev_copy, sizeof(lambertian), cudaMemcpyHostToDevice));
            data["material"][i]["device_ptr"] = reinterpret_cast<uintptr_t>(dev_material);
        }
        if (matdata["type"] == "metal") {
            color albedo = color(matdata["albedo"][0].get<double>(),
                                 matdata["albedo"][1].get<double>(),
                                 matdata["albedo"][2].get<double>());
            auto fuzz = matdata["fuzz"].get<double>();
            host_materials[i] = new metal(albedo, fuzz);
            data["material"][i]["host_ptr"] = reinterpret_cast<uintptr_t>(host_materials[i]);

            material *dev_material;
            checkCudaErrors(cudaMalloc((void **) &dev_material, sizeof(metal)));
            checkCudaErrors(cudaMemcpy(dev_material, host_materials[i], sizeof(metal), cudaMemcpyHostToDevice));
            data["material"][i]["device_ptr"] = reinterpret_cast<uintptr_t>(dev_material);
        }
        if (matdata["type"] == "dielectric") {
            auto ir = matdata["index_of_refraction"].get<double>();
            host_materials[i] = new dielectric(ir);
            data["material"][i]["host_ptr"] = reinterpret_cast<uintptr_t>(host_materials[i]);

            material *dev_material;
            checkCudaErrors(cudaMalloc((void **) &dev_material, sizeof(dielectric)));
            checkCudaErrors(cudaMemcpy(dev_material, host_materials[i], sizeof(dielectric), cudaMemcpyHostToDevice));
            data["material"][i]["device_ptr"] = reinterpret_cast<uintptr_t>(dev_material);
        }
    }

    material **dev_copy, **dev_materials;
    dev_copy = new material *[data["num_of_materials"]];
    for (int i = 0; i < data["num_of_materials"].get<int>(); ++i) {
        dev_copy[i] = reinterpret_cast<material *>(data["material"][i]["device_ptr"].get<uintptr_t>());
    }
    checkCudaErrors(cudaMalloc((void **) &dev_materials, sizeof(material *) * data["num_of_materials"].get<int>()));
    checkCudaErrors(cudaMemcpy(dev_materials, dev_copy, sizeof(material *) * data["num_of_materials"].get<int>(),
                               cudaMemcpyHostToDevice));

    data["materials"]["host_ptr"] = reinterpret_cast<uintptr_t>(host_materials);
    data["materials"]["device_ptr"] = reinterpret_cast<uintptr_t>(dev_materials);
    return host_materials;
}

hittable **parser_object(json &data) {
    hittable **host_objects = new hittable *[data["num_of_objects"]];
    for (int i = 0; i < data["num_of_objects"]; ++i) {
        auto objdata = data["object"][i];
        if (objdata["type"] == "sphere") {
            auto center = point3(objdata["center"][0].get<double>(),
                                 objdata["center"][1].get<double>(),
                                 objdata["center"][2].get<double>());
            auto radius = objdata["radius"].get<double>();
            auto material_id = objdata["material"].get<int>();
            host_objects[i] = new sphere(center, radius,
                                         reinterpret_cast<material *>(data["materials"][material_id]["host_ptr"].get<uintptr_t>()));
            data["object"][i]["host_ptr"] = reinterpret_cast<uintptr_t>(host_objects[i]);

            hittable *dev_copy, *dev_object;
            dev_copy = new sphere(center, radius,
                                  reinterpret_cast<material *>(data["materials"][material_id]["device_ptr"].get<uintptr_t>()));
            checkCudaErrors(cudaMalloc((void **) &dev_object, sizeof(sphere)));
            checkCudaErrors(cudaMemcpy(dev_object, dev_copy, sizeof(sphere), cudaMemcpyHostToDevice));
            data["object"][i]["device_ptr"] = reinterpret_cast<uintptr_t>(dev_object);
        }
    }

    hittable **dev_copy, **dev_objects;
    dev_copy = new hittable *[data["num_of_objects"]];
    for (int i = 0; i < data["num_of_objects"].get<int>(); ++i) {
        dev_copy[i] = reinterpret_cast<hittable *>(data["object"][i]["device_ptr"].get<uintptr_t>());
    }
    checkCudaErrors(cudaMalloc((void **) &dev_objects, sizeof(hittable *) * data["num_of_objects"].get<int>()));
    checkCudaErrors(cudaMemcpy(dev_objects, dev_copy, sizeof(hittable *) * data["num_of_objects"].get<int>(),
                               cudaMemcpyHostToDevice));

    data["objects"]["host_ptr"] = reinterpret_cast<uintptr_t>(host_objects);
    data["objects"]["device_ptr"] = reinterpret_cast<uintptr_t>(dev_objects);
    return host_objects;
}

hittable_list **parser_world(json &data) {
    hittable_list **host_world = new hittable_list *[1];
    host_world[0] = new hittable_list(reinterpret_cast<hittable **>(data["objects"]["host_ptr"].get<uintptr_t>()),
                                      data["num_of_objects"].get<int>());
    data["world"]["host_ptr"] = reinterpret_cast<uintptr_t>(host_world);

    hittable_list **dev_copy, **dev_world, dev_hittable_list(
            reinterpret_cast<hittable **>(data["objects"]["device_ptr"].get<uintptr_t>()),
            data["num_of_objects"].get<int>());
    dev_copy = new hittable_list *[1];
    checkCudaErrors(cudaMalloc((void **) dev_copy[0], sizeof(hittable_list)));
    checkCudaErrors(cudaMemcpy(dev_copy[0], &dev_hittable_list, sizeof(hittable_list), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &dev_world, sizeof(hittable_list *)));
    checkCudaErrors(cudaMemcpy(dev_world, dev_copy, sizeof(hittable_list *), cudaMemcpyHostToDevice));
    data["world"]["device_ptr"] = reinterpret_cast<uintptr_t>(dev_world);
    return host_world;
}

scene *parse_scene(const std::string &filename) {
    std::ifstream file(filename);
    json data = json::parse(file);

    auto *host_scene = new scene();

    host_scene->background = color(data["background"][0], data["background"][1], data["background"][2]);
    host_scene->max_depth = data["max_depth"];
    host_scene->samples_per_pixel = data["samples_per_pixel"];
    host_scene->width = data["width"];
    host_scene->height = data["height"];

    host_scene->num_of_objects = data["object"].size();
    host_scene->num_of_materials = data["material"].size();
    host_scene->num_of_textures = data["texture"].size();
    data["num_of_objects"] = host_scene->num_of_objects;
    data["num_of_materials"] = host_scene->num_of_materials;
    data["num_of_textures"] = host_scene->num_of_textures;

    when("num_of_objects: %d\n", host_scene->num_of_objects);
    when("num_of_materials: %d\n", host_scene->num_of_materials);
    when("num_of_textures: %d\n", host_scene->num_of_textures);

    // create camera on cpu
    host_scene->cam = parser_camera(data);

    // create textures on cpu
    host_scene->textures = parser_texture(data);

    // create materials on cpu
    host_scene->materials = parser_material(data);

    // create objects on cpu
    host_scene->objects = parser_object(data);

    // create world on cpu
    host_scene->world = parser_world(data);

    // move scene to device
    scene *dev_copy, *dev_scene;
    dev_copy = new scene();
    memcpy(dev_copy, host_scene, sizeof(scene));
    dev_copy->cam = reinterpret_cast<camera **>(data["camera"]["device_ptr"].get<uintptr_t>());
    dev_copy->textures = reinterpret_cast<mytexture **>(data["texture"]["device_ptr"].get<uintptr_t>());
    dev_copy->materials = reinterpret_cast<material **>(data["material"]["device_ptr"].get<uintptr_t>());
    dev_copy->objects = reinterpret_cast<hittable **>(data["object"]["device_ptr"].get<uintptr_t>());
    dev_copy->world = reinterpret_cast<hittable_list **>(data["world"]["device_ptr"].get<uintptr_t>());
    checkCudaErrors(cudaMalloc((void **) &dev_scene, sizeof(scene)));
    checkCudaErrors(cudaMemcpy(dev_scene, dev_copy, sizeof(scene), cudaMemcpyHostToDevice));

    data["host_ptr"] = reinterpret_cast<uintptr_t>(host_scene);
    data["device_ptr"] = reinterpret_cast<uintptr_t>(dev_scene);

    return dev_scene;
}