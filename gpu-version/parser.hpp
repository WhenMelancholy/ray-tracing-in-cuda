#pragma once

#include "material.cuh"
#include "vec3.cuh"
#include "rtweekend.cuh"
#include "texture.cuh"
#include "object.cuh"
#include "color.cuh"

#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct scene {
    int num_of_objects;
    int num_of_materials;
    int num_of_textures;
    hittable *objects;
    material *materials;
    mytexture *textures;

    hittable_list *world;
    camera *cam;
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

    // create objects on gpu
    return nullptr;
}