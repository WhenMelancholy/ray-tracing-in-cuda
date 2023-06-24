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
                           hittable *world, int depth, curandState *rng) {
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

        printf("Hit\n");
        hittable_list *world_list = (hittable_list *)world;
        printf("world_list: %p\n", world_list);
        printf("world_list->list: %d\n", world_list->len); // 正常工作
        // print the address of world->hit
        bool (hittable::*func_ptr)(const ray &, float, float, hit_record &)const  =
            &hittable::hit; // 获取函数指针
        printf("world->hit: %p\n", (void*)(world->*func_ptr));

        if (world->hit(now, 0.001, FLT_MAX, rec)) { // 直接崩溃
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

__global__ void render(int sample, camera *cam, hittable *world, int max_depth,
                       int image_width, int image_height, color *image,
                       curandState *states) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int id = y * image_width + x;

    if (x >= image_width)
        return;
    if (y >= image_height)
        return;
    if (x != 0)
        return;
    if (y != 0)
        return;

    curandState *rng = &states[id];

    color res(0, 0, 0);
    color background(0.70, 0.8, 1.0);
    for (int s = 0; s < sample; ++s) {
        //        printf("sample: %d/%d\n", s, sample);
        auto u = float(x + random_float(rng)) / (image_width - 1);
        auto v = float(y + random_float(rng)) / (image_height - 1);
        ray r = cam->get_ray(u, v, rng);
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
// UPDATE 使用了托管内存，不需要并行化生成了
void random_scene(hittable **list, hittable *&world, camera *&cam,
                  int image_width, int image_height, curandState *states,
                  int num_of_objects) {

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
        world = new hittable_list(list, 5);

        // Camera
        point3 lookfrom(13, 2, 3);
        point3 lookat(0, 0, 0);
        vec3 vup(0, 1, 0);
        auto dist_to_focus = (lookfrom - lookat).length();
        auto aperture = 0.1;
        //        *cam = new camera(lookfrom, lookat, vup, 20,
        //        float(image_width) / float(image_height), aperture,
        //        dist_to_focus);

        cam = new camera(vec3(-2, 2, 1), vec3(0, 0, -1), vec3(0, 1, 0), 20.0,
                         float(image_width) / float(image_height), 0,
                         dist_to_focus);

        return;
    }
}

int main(int argc, char *argv[]) {
    // cpu 计时功能
    auto start = clock();
    when("Start counting time\n");

    // 调整 cuda 堆栈大小
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 65536));

    // Init image
    constexpr auto aspect_ratio = 16.0 / 9.0;
    constexpr int image_width = 1600;
    constexpr int image_height = static_cast<int>(image_width / aspect_ratio);
    int max_depth = 50;
    int samples_per_pixel = 500;
    // const int num_of_objects = 22 * 22 + 1 + 3;
    const int num_of_objects = 5;

    // 根据命令行参数设置图像参数
    // UPDATE 删去调整图像长宽的参数
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0) {
            max_depth = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-spp") == 0) {
            samples_per_pixel = atoi(argv[i + 1]);
        }
    }

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

    // UPDATE hittable_list 需要从 vector
    // 迁移到数组，使用指针开辟空间，方便在显卡间传输数据 UPDATE hittable_list
    // 从数组迁移到 thrust_vector，数组不方便处理继承问题 UPDATE hittable_list
    // 还是使用了指针实现，并且在显卡上创建 在 cuda
    // 的函数中创建世界和相机，因为要使用 new 创建，不方便使用 malloc
    // 直接创建然后拷贝
    hittable **dev_lists, *dev_world;
    camera *dev_camera;
    checkCudaErrors(cudaMallocManaged((void **)&dev_lists,
                                      sizeof(hittable *) * num_of_objects));
    dev_world = nullptr;
    dev_camera = nullptr;
    when("Finish the allocation of objects, world, camera\n");

    random_scene(dev_lists, dev_world, dev_camera, image_width, image_height,
                 states, num_of_objects);
    when("Finish the creation of world, objects, camera\n");

    // 分配本地和显卡图像的空间
    static color image[num_of_pixels];
    color *dev_image;
    checkCudaErrors(
        cudaMalloc((void **)&dev_image, sizeof(color) * num_of_pixels));
    when("Finish the allocation of image\n");

    // 完成世界、相机、图像内存的初始化
    checkCudaErrors(cudaDeviceSynchronize());
    when("Start rendering\n");

    render<<<grids, threads>>>(samples_per_pixel, dev_camera, dev_world,
                               max_depth, image_width, image_height, dev_image,
                               states);
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
    // checkCudaErrors(cudaFree(dev_image));
    // checkCudaErrors(cudaFree(states));
    cudaDeviceReset();
}