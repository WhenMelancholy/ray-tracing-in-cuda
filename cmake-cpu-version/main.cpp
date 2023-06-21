#include "camera.h"
#include "hittable_list.h"
#include "material.h"
#include "rtweekend.h"
#include "sphere.h"

#include "color.h"

#include <iostream>
#include <cstdio>

// 计算光线 r 在 world 中的反射结果，最大深度为 depth
color ray_color(const ray& r, const hittable_list& world, int depth)
{
    ray now = r;
    color ret(1.0, 1.0, 1.0);
    // FIX 将 hittable 与 material 类整合到一起，方便数据传输
    // FIX 将递归调用改为循环判断，适应 cuda 的计算
    // FIX 仍然将 hittable 与 material 类分开，但是 hittable 在拷贝到显存后需要重新设置 material 的指针
    while (depth > 0) {
        hit_record rec;

        if (world.hit(now, 0.001, inf, rec)) {
            ray scattered;
            color attenuation;
            if (rec.mat_ptr->scatter(now, rec, attenuation, scattered))
            {
                ret = ret * attenuation;
                depth--;
                now = scattered;
                continue;
            }
            // 所有光线被吸收，返回黑色
            return color(0, 0, 0);
        }

        // 没有碰到物体，返回环境颜色
        vec3 unit_dir = unit_vector(now.direction());
        auto t = 0.5 * (unit_dir.y() + 1.0);
        return ((1.0 - t) * color(1.0, 1, 1) + t * color(0.5, 0.7, 1.0))*ret;
    }
 
    // 超过最大深度，光线衰减到 0
    return color(0, 0, 0);
}

color render(double x, double y, int sample, camera cam, hittable_list world, int max_depth,int image_width,int image_height) {
    color res;
    for (int s = 0; s < sample; ++s) {
        auto u = double(x + random_double()) / (image_width - 1);
        auto v = double(y + random_double()) / (image_height - 1);
        ray r = cam.get_ray(u, v);
        res += ray_color(r, world, max_depth);
    }
    return res;
}

hittable_list random_scene();
void run(int argc, char* argv[], long start) {
    // 重定向输出到 main.ppm 
    (void)freopen("main.ppm", "w", stdout);

    // Init image
    const auto aspect_ratio = 16.0 / 9.0;
    int image_width = 400;
    int image_height = static_cast<int>(image_width / aspect_ratio);
    int max_depth = 50;
    int samples_per_pixel = 50;

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

    // FIX hittable_list 需要从 vector 迁移到数组，使用指针开辟空间，方便在显卡间传输数据
    // FIX hittable_list 从数组迁移到 thrust_vector，数组传输不方便处理继承问题
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
            color c=render(i,j,samples_per_pixel,cam,world,max_depth,image_width,image_height);
            write_color(std::cout, c, samples_per_pixel);
        }
    }

    fprintf(stderr, "\nDone\n");

    FILE* fp = fopen("gpu-version-time.log", "a");
    fprintf(fp, "basic cpu verion, image width: %d,image height: %d, max depth: %d, samples per pixel: %d, time: %lf s\n",
        image_width, image_height, max_depth, samples_per_pixel, (clock() - start) / double(CLOCKS_PER_SEC));
    fclose(fp);
}

int main(int argc,char *argv[]) {
    // 使用固定的随机数种子，保证每次测试生成的物体和发出的光线是一样的
    srand(7);

    for (int i = 0; i < 1; ++i)
        run(argc, argv, clock());
}

hittable_list random_scene() {
    hittable_list world;

    auto ground_material = new lambertian(color(0.5, 0.5, 0.5));
    world.add(new sphere(point3(0, -1000, 0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                material* sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = new lambertian(albedo);
                    world.add(new sphere(center, 0.2, sphere_material));
                }
                else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = new metal(albedo, fuzz);
                    world.add(new sphere(center, 0.2, sphere_material));
                }
                else {
                    // glass
                    sphere_material = new dielectric(1.5);
                    world.add(new sphere(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = new dielectric(1.5);
    world.add(new sphere(point3(0, 1, 0), 1.0, material1));

    auto material2 = new lambertian(color(0.4, 0.2, 0.1));
    world.add(new sphere(point3(-4, 1, 0), 1.0, material2));

    auto material3 = new metal(color(0.7, 0.6, 0.5), 0.0);
    world.add(new sphere(point3(4, 1, 0), 1.0, material3));

    return world;
}