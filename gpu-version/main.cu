#include "camera.cuh"
#include "object.cuh"
#include "material.cuh"
#include "rtweekend.cuh"
#include "texture.cuh"

#include "color.cuh"
#include "parser.hpp"

#include <iostream>
#include <cfloat>
#include <cstdio>
#include <cuda.h>
#include <curand_kernel.h>

// 计算光线 r 在 world 中的反射结果，最大深度为 depth
__device__ color ray_color(const ray &r, const color &background,
                           hittable **world, int depth, curandState *rng) {
    //    UPDATE: 改为递归运算，因为非递归需要用栈来合并最终结果
    ray now = r;
    color accumulated_attenuation(1.0f, 1.0f, 1.0f);
    color accumulated_color(0.0f, 0.0f, 0.0f);
    // UPDATE 将 hittable 与 material 类整合到一起，方便数据传输
    // UPDATE 将递归调用改为循环判断，适应 cuda 的计算
    // UPDATE 仍然将 hittable 与 material 类分开，但是 hittable
    // 在拷贝到显存后需要重新设置 material 的指针
    while (depth > 0) {
        hit_record rec;

        // hittable_list *list = (hittable_list *)(*world);
        // if (list != nullptr) {
        //     printf("successful cast to hittable_list\n");
        //     printf("list size: %d\n", list->len);
        //     sphere *s = (sphere *)list->objects[0];
        //     printf("sphere center: %f %f %f\n", s->center.x(), s->center.y(),
        //            s->center.z());
        //     lambertian *lam = (lambertian *)s->mat_ptr;
        //     solid_color *sc = (solid_color *)lam->albedo;
        //     printf("lambertian albedo: %f %f %f\n", sc->color_value.x(),
        //            sc->color_value.y(), sc->color_value.z());
        // } else {
        //     printf("failed cast to hittable_list\n");
        // }

        if ((*world)->hit(now, 0.001, FLT_MAX, rec)) {
            ray scattered;
            color attenuation;
            color emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);

            if (rec.mat_ptr->scatter(now, rec, attenuation, scattered, rng)) {
                accumulated_color += emitted * accumulated_attenuation;
                accumulated_attenuation = accumulated_attenuation * attenuation;
                depth--;
                now = scattered;
                continue;
            } else {
                // 没有反射，返回自发光
                accumulated_color += accumulated_attenuation * emitted;
                break;
            }
        } else {
            // 没有碰到物体，返回环境颜色
            accumulated_color += accumulated_attenuation * background;
            break;
        }
    }

    // 超过最大深度，光线衰减到 0
    return accumulated_color;
}

__global__ void render(int sample, color background, camera **cam,
                       hittable **world, int max_depth, int image_width,
                       int image_height, color *image, curandState *states) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int id = y * image_width + x;

    if (x >= image_width)
        return;
    if (y >= image_height)
        return;

    curandState *rng = &states[id];

    color res(0, 0, 0);

    // print detail info of cam
    // printf("%d\n", __LINE__);
    // auto cam_data = **cam;
    // printf("%d\n", __LINE__);
    // printf("cam: %f %f %f\n", cam_data.origin.x(), cam_data.origin.y(),
    //        cam_data.origin.z());

    for (int s = 0; s < sample; ++s) {
        //        printf("sample: %d/%d\n", s, sample);
        auto u = float(x + random_float(rng)) / (image_width - 1);
        auto v = float(y + random_float(rng)) / (image_height - 1);
        ray r = (*cam)->get_ray(u, v, rng);
        res += ray_color(r, background, world, max_depth, rng);
    }
    // UPDATE 将除以采样数的操作移动到了 kernel 函数内
    // UPDATE 还是将操作保留在 write_color 函数里吧
    image[id] = res;
}

// UPDATE 复制 PI 的全局变量和 inf 的全局变量到设备内存
void init_constant() {
    constexpr float tmp_inf = std::numeric_limits<float>::infinity();
    //    constexpr float tmp_inf = 1e9;
    const float tmp_pi = acos(-1);
    when("inf: %f, pi: %f\n", tmp_inf, tmp_pi);

    // UPDATE cudaMemcpyToSymbol 中设备端的变量是不需要加 & 的
    // UPDATE 使用 define 定义的常量替代常数常量
    // checkCudaErrors(cudaMemcpyToSymbol(inf, &tmp_inf, sizeof(float)));
    // checkCudaErrors(cudaMemcpyToSymbol(pi, &tmp_pi, sizeof(float)));
}

__global__ void init_random_library(curandState *state) {
    int idx = blockIdx.x;
    // 固定种子，方便比较性能
    // UPDATE 更新随机数生成方法
    curand_init(idx, 0, 0, &state[idx]);
}

// UPDATE 并行化世界生成
__global__ void random_scene(hittable **list, hittable **world, camera **cam,
                             int image_width, int image_height,
                             curandState *states, int num_of_objects) {

    //     UPDATE 添加小型场景进行测试
    if (true) {
        list[0] = new sphere(vec3(0, 0, -1), 0.5,
                             new lambertian(vec3(0.1, 0.2, 0.5)));
        list[1] = new sphere(vec3(0, -100.5, -1), 100,
                             new lambertian(vec3(0.8, 0.8, 0.0)));
        list[2] = new sphere(vec3(1, 0, -1), 0.5,
                             new metal(vec3(0.8, 0.6, 0.2), 0.0));
        list[3] = new sphere(vec3(-1, 0, -1), 0.5, new dielectric(1.5));
        list[4] = new sphere(vec3(-1, 0, -1), -0.45, new dielectric(1.5));
        *world = new hittable_list(list, 5);

        // Camera
        point3 lookfrom(13, 2, 3);
        point3 lookat(0, 0, 0);
        vec3 vup(0, 1, 0);
        auto dist_to_focus = (lookfrom - lookat).length();
        //        *cam = new camera(lookfrom, lookat, vup, 20,
        //        float(image_width) / float(image_height), 0.1,
        //        dist_to_focus);

        *cam = new camera(vec3(-2, 2, 1), vec3(0, 0, -1), vec3(0, 1, 0), 20.0,
                          float(image_width) / float(image_height), 0,
                          dist_to_focus);

        return;
    }

    int id = blockIdx.x;
    auto *rng = &states[id];

    int a = id / 22 - 11;
    int b = id % 22 - 11;
    float choose_mat = random_float(rng);
    vec3 center(a + 0.9 * random_float(rng), 0.2, b + 0.9 * random_float(rng));

    material *sphere_material;

    // 为了保证数量固定，球只要生成了就会加入到世界
    if (choose_mat < 0.8f) {
        auto albedo = color::random(rng) * color::random(rng);
        sphere_material = new lambertian(albedo);
        list[id] = new sphere(center, 0.2, sphere_material);
    } else if (choose_mat < 0.95f) {
        auto albedo = color::random(0.5, 1, rng);
        auto fuzz = random_float(rng);
        sphere_material = new metal(albedo, fuzz);
        list[id] = new sphere(center, 0.2, sphere_material);
    } else {
        sphere_material = new dielectric(1.5);
        list[id] = new sphere(center, 0.2, sphere_material);
    }

    if (id == 0) {
        //        list[num_of_objects - 4] = new sphere(vec3(0, -1000.0, 0),
        //        1000, new lambertian(vec3(0.5, 0.5, 0.5)));
        list[num_of_objects - 4] =
            new sphere(vec3(0, -1000.0, 0), 1000,
                       new lambertian(new checker_texture(
                           color(0.2, 0.3, 0.1), color(0.9, 0.9, 0.9))));
        list[num_of_objects - 3] =
            new sphere(vec3(0, 2, 0), 1.0, new dielectric(1.5));
        list[num_of_objects - 2] = new sphere(
            vec3(-4, 2, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));

        auto difflight = new diffuse_light(color(4, 4, 4));
        auto rect_light = new xy_rect(3, 5, 1, 3, -2, difflight);
        //        list[num_of_objects - 1] = new sphere(vec3(4, 2, 0), 1.0, new
        //        metal(vec3(0.7, 0.6, 0.5), 0.0));;
        auto cylinder_light = new cylinder(0.5, 0, 2, difflight);
        list[num_of_objects - 1] = cylinder_light;

        *world = new hittable_list(list, num_of_objects);

        // Camera
        point3 lookfrom(2, 2, -13);
        point3 lookat(0, 0, 0);
        vec3 vup(0, 1, 0);
        auto dist_to_focus = (lookfrom - lookat).length();
        auto aperture = 0.1;
        *cam = new camera(lookfrom, lookat, vup, 20,
                          float(image_width) / float(image_height), aperture,
                          dist_to_focus);
    }
}

__global__ void free_scene(hittable **list, hittable **world, camera **cam,
                           int num_of_objects) {
    for (int i = 0; i < num_of_objects; i++) {
        delete list[i];
    }
    delete *world;
    delete *cam;
}

std::tuple<hittable **, camera **> get_coded_scene(int image_width,
                                                   int image_height,
                                                   int num_of_objects,
                                                   curandState *states) {
    // UPDATE hittable_list 需要从 vector
    // 迁移到数组，使用指针开辟空间，方便在显卡间传输数据 UPDATE hittable_list
    // 从数组迁移到 thrust_vector，数组不方便处理继承问题 UPDATE hittable_list
    // 还是使用了指针实现，并且在显卡上创建 在 cuda
    // 的函数中创建世界和相机，因为要使用 new 创建，不方便使用 malloc
    // 直接创建然后拷贝
    hittable **dev_lists, **dev_world;
    camera **dev_camera;

    checkCudaErrors(
        cudaMalloc((void **)&dev_lists, sizeof(hittable *) * num_of_objects));
    checkCudaErrors(cudaMalloc((void **)&dev_world, sizeof(hittable *)));
    checkCudaErrors(cudaMalloc((void **)&dev_camera, sizeof(camera *)));
    when("Finish the allocation of objects, world, camera\n");

    random_scene<<<num_of_objects, 1>>>(dev_lists, dev_world, dev_camera,
                                        image_width, image_height, states,
                                        num_of_objects);
    when("Finish the creation of world, objects, camera\n");

    return {dev_world, dev_camera};
}
int oldmain(int argc, char *argv[]) {
    // cpu 计时功能
    auto start = clock();
    when("Start counting time\n");

    // Init image
    constexpr auto aspect_ratio = 16.0 / 9.0;
    constexpr int image_width = 800;
    constexpr int image_height = static_cast<int>(image_width / aspect_ratio);
    int samples_per_pixel = 500;
    const int num_of_objects = 5;

    int max_depth = 50;

    // 根据命令行参数设置图像参数
    // UPDATE 删去调整图像长宽的参数
    // for (int i = 0; i < argc; i++) {
    //     if (strcmp(argv[i], "-d") == 0) {
    //         max_depth = atoi(argv[i + 1]);
    //     } else if (strcmp(argv[i], "-spp") == 0) {
    //         samples_per_pixel = atoi(argv[i + 1]);
    //     }
    // }

    const int wrap = 8;
    dim3 grids(image_width / wrap + 1, image_height / wrap + 1);
    dim3 threads(wrap, wrap);

    // 随机化库的初始化操作
    curandStateXORWOW_t *states;
    constexpr int num_of_pixels = image_height * image_width;
    checkCudaErrors(
        cudaMalloc(&states, sizeof(curandStateXORWOW_t) * num_of_pixels));
    when("Finish the memory allocation of random library\n");

    // 随机数生成器的初始化操作
    // UPDATE 将随机数初始化从 1xnum_of_pixels 改为
    // num_of_pixelsx1，前者会超过线程数限制
    init_constant();
    init_random_library<<<num_of_pixels, 1>>>(states);

    // 完成随机数库和常数的初始化
    checkCudaErrors(cudaDeviceSynchronize());
    when("Finish the initialization of random library and constants\n");

    // UPDATE 将世界和相机的创建放到函数中
    hittable **dev_world;
    camera **dev_camera;
    std::tie(dev_world, dev_camera) =
        get_coded_scene(image_width, image_height, num_of_objects, states);

    // 分配本地和显卡图像的空间
    static color image[num_of_pixels];
    color *dev_image;
    checkCudaErrors(
        cudaMalloc((void **)&dev_image, sizeof(color) * num_of_pixels));
    when("Finish the allocation of image\n");

    // 完成世界、相机、图像内存的初始化
    checkCudaErrors(cudaDeviceSynchronize());
    when("Start rendering\n");

    render<<<grids, threads>>>(samples_per_pixel, color(0.3, 0.7, 1.0),
                               dev_camera, dev_world, max_depth, image_width,
                               image_height, dev_image, states);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    when("Finish rendering\n");

    // 输出
    checkCudaErrors(cudaMemcpy(image, dev_image, sizeof(color) * num_of_pixels,
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    when("Copying image\n");

    // 重定向输出到 main.ppm
    (void)freopen("main.ppm", "w", stdout);
    printf("P3\n%d %d\n255\n", image_width, image_height);

    for (int j = image_height - 1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {
            write_color(stdout, image[j * image_width + i], samples_per_pixel);
        }
    }

    FILE *fp = fopen("gpu-cuda-version-time.log", "a");
    fprintf(
        fp,
        "basic cuda versions, improve world generate, image width: %d,image "
        "height: %d, max depth: %d, samples per pixel: %d, time: %f s\n",
        image_width, image_height, max_depth, samples_per_pixel,
        (clock() - start) / float(CLOCKS_PER_SEC));
    fclose(fp);

    // 清理退出程序
    // UPDATE 让操作系统去 free 把，free 不动了
    // free_scene<<<1, 1>>>(dev_lists, dev_world, dev_camera, num_of_objects);
    // checkCudaErrors(cudaFree(dev_lists));
    // checkCudaErrors(cudaFree(dev_world));
    // checkCudaErrors(cudaFree(dev_camera));
    // checkCudaErrors(cudaFree(dev_image));
    // checkCudaErrors(cudaFree(states));
    cudaDeviceReset();
    return 0;
}

void output_image(color *image, int image_width, int image_height,
                  int samples_per_pixel, std::string filename) {
    // 重定向输出到 main.ppm
    FILE *fp = fopen(filename.c_str(), "w");
    fprintf(fp, "P3\n%d %d\n255\n", image_width, image_height);

    for (int j = image_height - 1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {
            write_color(fp, image[j * image_width + i], samples_per_pixel);
        }
    }

    fclose(fp);
}

__device__ mytexture *move_to_device(mytexture *src) {
    if (src->type == class_type::solid_color) {
        return new solid_color(((solid_color *)src)->color_value);
    }
    if (src->type == class_type::checker) {
        return new checker_texture(
            move_to_device(((checker_texture *)src)->odd),
            move_to_device(((checker_texture *)src)->even));
    }
    printf("error happend in %s:%d\n", __FILE__, __LINE__);
}

__device__ material *move_to_device(material *src) {
    if (src->type == class_type::lambertian) {
        return new lambertian(move_to_device(((lambertian *)src)->albedo));
    }
    if (src->type == class_type::metal) {
        return new metal(((metal *)src)->albedo, ((metal *)src)->fuzz);
    }
    if (src->type == class_type::dielectric) {
        return new dielectric(((dielectric *)src)->ir);
    }
    if (src->type == class_type::diffuse_light) {
        return new diffuse_light(move_to_device(((diffuse_light *)src)->emit));
    }
    printf("error happend in %s:%d\n", __FILE__, __LINE__);
}

__device__ hittable *move_to_device(hittable *src) {
    if (src->type == class_type::xy_rect) {
        auto tmp = (xy_rect *)src;
        return new xy_rect(tmp->x0, tmp->x1, tmp->y0, tmp->y1, tmp->k,
                           move_to_device(tmp->mp));
    }
    if (src->type == class_type::yz_rect) {
        auto tmp = (yz_rect *)src;
        return new yz_rect(tmp->y0, tmp->y1, tmp->z0, tmp->z1, tmp->k,
                           move_to_device(tmp->mp));
    }
    if (src->type == class_type::xz_rect) {
        auto tmp = (xz_rect *)src;
        return new xz_rect(tmp->x0, tmp->x1, tmp->z0, tmp->z1, tmp->k,
                           move_to_device(tmp->mp));
    }
    if (src->type == class_type::hittable_list) {
        hittable_list *dst = new hittable_list();
        dst->len = ((hittable_list *)src)->len;
        dst->objects = new hittable *[dst->len];
        for (int i = 0; i < dst->len; i++) {
            dst->objects[i] =
                move_to_device(((hittable_list *)src)->objects[i]);
        }
        // return new bvh_node(dst->objects, 0, dst->len);
        return dst;
    }
    if (src->type == class_type::sphere) {
        return new sphere(((sphere *)src)->center, ((sphere *)src)->radius,
                          move_to_device(((sphere *)src)->mat_ptr));
    }
    if (src->type == class_type::cylinder) {
        auto dev_obj =
            new cylinder(((cylinder *)src)->radius, ((cylinder *)src)->zmin,
                         ((cylinder *)src)->zmax,
                         move_to_device(((cylinder *)src)->mat_ptr));
        dev_obj->o2w = ((cylinder *)src)->o2w;
        return dev_obj;
    }
    printf("error happend in %s:%d\n", __FILE__, __LINE__);
}

__global__ void move_to_device(hittable **src, hittable **dst) {
    *dst = move_to_device(*src);
}

int jsonmain(int argc, char *argv[]) {
    cudaDeviceSetLimit(cudaLimitStackSize, 8192 * 4);
    // cpu 计时功能
    auto start = clock();
    when("Start counting time\n");

    // parse scene file name from argc
    std::string scene_file_name = "sample_scene.json";
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "-f") == 0) {
            scene_file_name = argv[i + 1];
        }
    }

    scene *world;
    json data;
    std::tie(world, data) = parse_scene(scene_file_name);
    when("Finish parsing scene\n");

    const int wrap = 8;
    dim3 grids(world->width / wrap + 1, world->height / wrap + 1);
    dim3 threads(wrap, wrap);

    hittable **dev_world;
    checkCudaErrors(cudaMalloc((void **)&dev_world, sizeof(hittable *)));
    move_to_device<<<1, 1>>>(world->world, dev_world);
    when("Finish the allocation of objects, world, camera\n");

    auto dev_camera = world->cam;
    when("Finish the allocation of objects, world, camera\n");

    // 分配本地和显卡图像的空间
    int num_of_pixels = world->height * world->width;
    color *image;
    checkCudaErrors(
        cudaMallocManaged((void **)&image, sizeof(color) * num_of_pixels));
    when("Finish the allocation of image\n");

    // 随机化库的初始化操作
    curandStateXORWOW_t *states;
    checkCudaErrors(
        cudaMalloc(&states, sizeof(curandStateXORWOW_t) * num_of_pixels));
    when("Finish the memory allocation of random library\n");

    // 随机数生成器的初始化操作
    // UPDATE 将随机数初始化从 1xnum_of_pixels 改为
    // num_of_pixelsx1，前者会超过线程数限制
    init_constant();
    init_random_library<<<num_of_pixels, 1>>>(states);

    // 完成随机数库和常数的初始化
    checkCudaErrors(cudaDeviceSynchronize());
    when("Finish the initialization of random library and constants\n");

    checkCudaErrors(cudaDeviceSynchronize());
    when("Start rendering\n");

    render<<<grids, threads>>>(world->samples_per_pixel, world->background,
                               dev_camera, dev_world, world->max_depth,
                               world->width, world->height, image, states);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    when("Finish rendering\n");

    output_image(image, world->width, world->height, world->samples_per_pixel,
                 "main.ppm");
    write_image(world->width, world->height, image, world->samples_per_pixel,
                data["output_file"].get<std::string>());
    when("Finish writing image\n");

    cudaDeviceReset();
    when("Program finish, cost: %f s\n",
         double(clock() - start) / CLOCKS_PER_SEC);
    return 0;
}

int main(int argc, char *argv[]) {
    // oldmain(argc,argv);
    jsonmain(argc, argv);
}