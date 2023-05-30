#include "camera.h"
#include "hittable_list.h"
#include "material.h"
#include "rtweekend.h"
#include "sphere.h"

#include "color.h"
#include <iostream>
#include <cstdio>
#include <vector>

color ray_color(const ray& r, const hittable_list& world, int depth)
{
    hit_record rec;

    if (depth <= 0)
        return color(0, 0, 0);

    if (world.hit(r, 0.001, inf, rec)) {
        ray scattered;
        color attenuation;
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
            return attenuation * (ray_color(scattered, world, depth - 1));
        return color(0, 0, 0);
    }

    vec3 unit_dir = unit_vector(r.direction());
    auto t = 0.5 * (unit_dir.y() + 1.0);
    return (1.0 - t) * color(1.0, 1, 1) + t * color(0.5, 0.7, 1.0);
}

// 随机生成场景
hittable_list random_scene();

void run(int argc,char *argv[],long start) {
    // 输出到 main.ppm 
    (void)freopen("main.ppm", "w", stdout);

    // Init image
    const auto aspect_ratio = 16.0 / 9.0;
    int image_width = 400;
    int image_height = static_cast<int>(image_width / aspect_ratio);
    int max_depth = 50;
    int samples_per_pixel = 500;

    // 根据命令行参数设置图像参数
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "-w") == 0) {
            image_width = atoi(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-h") == 0) {
            image_height = atoi(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-d") == 0) {
            max_depth = atoi(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-spp") == 0) {
            samples_per_pixel = atoi(argv[i + 1]);
        }
    }

    // world
    hittable_list world = random_scene();

    // Camera
    point3 lookfrom(13, 2, 3);
    point3 lookat(0, 0, 0);
    vec3 vup(0, 1, 0);
    auto dist_to_focus = (lookfrom - lookat).length();
    auto aperture = 0.1;
    camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);

    // render
    printf("P3\n%d %d\n255\n", image_width, image_height);

    for (int j = image_height - 1; j >= 0; --j) {
        fprintf(stderr, "\rScanlines remaining %d ", j);
        fflush(stderr);
        for (int i = 0; i < image_width; ++i) {
            color c(0, 0, 0);
            for (int s = 0; s < samples_per_pixel; ++s) {
                auto u = double(i + random_double()) / (image_width - 1);
                auto v = double(j + random_double()) / (image_height - 1);
                ray r = cam.get_ray(u, v);
                c += ray_color(r, world, max_depth);
            }
            write_color(std::cout, c, samples_per_pixel);
        }
    }

    fprintf(stderr, "\nDone\n");

    FILE* fp = fopen("cpu-version-time.log", "a");
    fprintf(fp, "cpu verion, image width: %d,image height: %d, max depth: %d, samples per pixel: %d, time: %lf s\n",
        image_width, image_height, max_depth, samples_per_pixel, (clock()-start)/double(CLOCKS_PER_SEC));
    fclose(fp);
}

void omp_run(int argc,char *argv[],long start) {
    // 输出到 main.ppm 
    (void)freopen("main.ppm", "w", stdout);

    // Init image
    const auto aspect_ratio = 16.0 / 9.0;
    int image_width = 400;
    int image_height = static_cast<int>(image_width / aspect_ratio);
    int max_depth = 50;
    int samples_per_pixel = 500;

    // 根据命令行参数设置图像参数
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "-w") == 0) {
            image_width = atoi(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-h") == 0) {
            image_height = atoi(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-d") == 0) {
            max_depth = atoi(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-spp") == 0) {
            samples_per_pixel = atoi(argv[i + 1]);
        }
    }

    // world
    hittable_list world = random_scene();

    // Camera
    point3 lookfrom(13, 2, 3);
    point3 lookat(0, 0, 0);
    vec3 vup(0, 1, 0);
    auto dist_to_focus = (lookfrom - lookat).length();
    auto aperture = 0.1;
    camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);

    // render
    printf("P3\n%d %d\n255\n", image_width, image_height);

    std::vector<std::vector<color>> image;
    for (int i = 0; i < image_height; ++i) {
        image.push_back(std::vector<color>());
        for (int j = 0; j < image_width; ++j)
            image[i].push_back(color(0,0,0));
    }

    #pragma omp parallel for
    for (int j = image_height - 1; j >= 0; --j) {
        fprintf(stderr, "\rScanlines remaining %d ", j);
        fflush(stderr);
        for (int i = 0; i < image_width; ++i) {
            color c(0, 0, 0);
            for (int s = 0; s < samples_per_pixel; ++s) {
                auto u = double(i + random_double()) / (image_width - 1);
                auto v = double(j + random_double()) / (image_height - 1);
                ray r = cam.get_ray(u, v);
                c += ray_color(r, world, max_depth);
            }
            image[j][i] = c;
        }
    }
    for (int j = image_height - 1; ~j; --j)
        for (int i = 0; i < image_width; ++i)
            write_color(std::cout, image[j][i], samples_per_pixel);

    fprintf(stderr, "\nDone\n");

    FILE* fp = fopen("cpu-version-time.log", "a");
    fprintf(fp, "oepnmp second verion, image width: %d,image height: %d, max depth: %d, samples per pixel: %d, time: %lf s\n",
        image_width, image_height, max_depth, samples_per_pixel, (clock() - start) / double(CLOCKS_PER_SEC));
    fclose(fp);
}

// 在有电源的情况下，使用固定的随机数比较 cpu 版本和 openmp 版本的性能
// 将 openmp 的循环移至最外层以获得更好的效果
#define OMP_VERSION
#define CPU_VERSION
int main(int argc, char* argv[])
{
    srand(7);
#ifdef CPU_VERSION
    for (int i = 0; i < 1; ++i) {
        run(argc, argv, clock());
    }
#endif
#ifdef OMP_VERSION
    for (int i = 0; i < 1; ++i) {
        omp_run(argc, argv, clock());
    }
#endif
}

hittable_list random_scene()
{
    hittable_list world;

    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
                else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
                else {
                    // glass
                    sphere_material = make_shared<dielectric>(1.5);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    return world;
}
