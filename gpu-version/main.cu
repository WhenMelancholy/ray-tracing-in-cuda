#include "camera.cuh"
#include "hittable_list.cuh"
#include "material.cuh"
#include "rtweekend.cuh"
#include "sphere.cuh"
#include <assert.h>

#include "color.cuh"

#include <iostream>
#include <cstdio>
#include <cuda.h>
#include <curand_kernel.h>

// debug �������
#define when(...) fprintf(stderr,__VA_ARGS__)
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                  file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// ������� r �� world �еķ�������������Ϊ depth
__device__ color ray_color(const ray &r, hittable **world, int depth, curandState *rng) {
    ray now = r;
    color ret(1.0f, 1.0f, 1.0f);
    // UPDATE �� hittable �� material �����ϵ�һ�𣬷������ݴ���
    // UPDATE ���ݹ���ø�Ϊѭ���жϣ���Ӧ cuda �ļ���
    // UPDATE ��Ȼ�� hittable �� material ��ֿ������� hittable �ڿ������Դ����Ҫ�������� material ��ָ��
    while (depth > 0) {
        hit_record rec;

        if ((*world)->hit(now, 0.001, FLT_MAX, rec)) {
            ray scattered;
            color attenuation;

            if (rec.mat_ptr->scatter(now, rec, attenuation, scattered, rng)) {
                ret = ret * attenuation;
                depth--;
                now = scattered;
                continue;
            }
            // ���й��߱����գ����غ�ɫ
            return color(0, 0, 0);
        }

        // û���������壬���ػ�����ɫ
        vec3 unit_dir = unit_vector(now.direction());
        auto t = 0.5 * (unit_dir.y() + 1.0);
        return ((1.0 - t) * color(1.0, 1, 1) + t * color(0.5, 0.7, 1.0)) * ret;
    }

    // ���������ȣ�����˥���� 0
    return color(0, 0, 0);
}

__global__ void
render(int sample, camera **cam, hittable **world, int max_depth, int image_width, int image_height, color *image,
       curandState *states) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int id = y * image_width + x;

    if (x >= image_width) return;
    if (y >= image_height) return;

    curandState *rng = &states[id];

    color res(0, 0, 0);
    for (int s = 0; s < sample; ++s) {
        auto u = float(x + random_float(rng)) / (image_width - 1);
        auto v = float(y + random_float(rng)) / (image_height - 1);
        ray r = (*cam)->get_ray(u, v, rng);
        res += ray_color(r, world, max_depth, rng);
    }
    // UPDATE �����Բ������Ĳ����ƶ����� kernel ������
    // UPDATE ���ǽ����������� write_color �������
    image[id] = res;
}

// UPDATE ���� PI ��ȫ�ֱ����� inf ��ȫ�ֱ������豸�ڴ�
void init_constant() {
    constexpr float tmp_inf = std::numeric_limits<float>::infinity();
//    constexpr float tmp_inf = 1e9;
    const float tmp_pi = acos(-1);
    when("inf: %f, pi: %f\n", tmp_inf, tmp_pi);

    // UPDATE cudaMemcpyToSymbol ���豸�˵ı����ǲ���Ҫ�� & ��
    // UPDATE ʹ�� define ����ĳ��������������
//    checkCudaErrors(cudaMemcpyToSymbol(inf, &tmp_inf, sizeof(float)));
//    checkCudaErrors(cudaMemcpyToSymbol(pi, &tmp_pi, sizeof(float)));
}

__global__ void init_random_library(curandState *state) {
    int idx = blockIdx.x;
    // �̶����ӣ�����Ƚ�����
    // UPDATE ������������ɷ���
    curand_init(idx, 0, 0, &state[idx]);
}

// UPDATE ���л���������
__global__ void
random_scene(hittable **list, hittable **world, camera **cam, int image_width, int image_height, curandState *states,
             int num_of_objects) {

    // UPDATE ���С�ͳ������в���
//    if (false) {
//        list[0] = new sphere(vec3(0, 0, -1), 0.5,
//                             new lambertian(vec3(0.1, 0.2, 0.5)));
//        list[1] = new sphere(vec3(0, -100.5, -1), 100,
//                             new lambertian(vec3(0.8, 0.8, 0.0)));
//        list[2] = new sphere(vec3(1, 0, -1), 0.5,
//                             new metal(vec3(0.8, 0.6, 0.2), 0.0));
//        list[3] = new sphere(vec3(-1, 0, -1), 0.5,
//                             new dielectric(1.5));
//        list[4] = new sphere(vec3(-1, 0, -1), -0.45,
//                             new dielectric(1.5));
//        *world = new hittable_list(list, 5);
//
//        // Camera
//        point3 lookfrom(13, 2, 3);
//        point3 lookat(0, 0, 0);
//        vec3 vup(0, 1, 0);
//        auto dist_to_focus = (lookfrom - lookat).length();
//        auto aperture = 0.1;
////    *cam = new camera(lookfrom, lookat, vup, 20, float(image_width) / float(image_height), aperture, dist_to_focus);
//
//        *cam = new camera(vec3(-2, 2, 1),
//                          vec3(0, 0, -1),
//                          vec3(0, 1, 0),
//                          20.0,
//                          float(image_width) / float(image_height), 0, dist_to_focus);
//
//        return;
//    }

    int id = blockIdx.x;
    auto *rng = &states[id];

    int a = id / 22 - 11;
    int b = id % 22 - 11;
    float choose_mat = random_float(rng);
    vec3 center(a + 0.9 * random_float(rng), 0.2, b + 0.9 * random_float(rng));

    material *sphere_material;

    // Ϊ�˱�֤�����̶�����ֻҪ�����˾ͻ���뵽����
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
        list[num_of_objects - 4] = new sphere(vec3(0, -1000.0, 0), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));
        list[num_of_objects - 3] = new sphere(vec3(0, 2, 0), 1.0, new dielectric(1.5));
        list[num_of_objects - 2] = new sphere(vec3(-4, 2, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        list[num_of_objects - 1] = new sphere(vec3(4, 2, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));;

        *world = new hittable_list(list, num_of_objects);

        // Camera
        point3 lookfrom(13, 2, 3);
        point3 lookat(0, 0, 0);
        vec3 vup(0, 1, 0);
        auto dist_to_focus = (lookfrom - lookat).length();
        auto aperture = 0.1;
        *cam = new camera(lookfrom, lookat, vup, 20, float(image_width) / float(image_height), aperture, dist_to_focus);
    }
}

__global__ void free_scene(hittable **list, hittable **world, camera **cam, int num_of_objects) {
    for (int i = 0; i < num_of_objects; i++) {
        delete ((sphere *) list[i])->mat_ptr;
        delete list[i];
    }
    delete *world;
    delete *cam;
}


int main(int argc, char *argv[]) {
    // cpu ��ʱ����
    auto start = clock();
    when("��ʼ��ʱ\n");

    // �ض�������� main.ppm
    (void) freopen("main.ppm", "w", stdout);

    // Init image
    constexpr auto aspect_ratio = 16.0 / 9.0;
    constexpr int image_width = 400;
    constexpr int image_height = static_cast<int>(image_width / aspect_ratio);
    int max_depth = 50;
    int samples_per_pixel = 500;
    const int num_of_objects = 22 * 22 + 1 + 3;
//    const int num_of_objects = 3;

    // ���������в�������ͼ�����
    // UPDATE ɾȥ����ͼ�񳤿�Ĳ���
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

    // �������ĳ�ʼ������
    curandStateXORWOW_t *states;
    constexpr int num_of_pixels = image_height * image_width;
    checkCudaErrors(cudaMalloc(&states, sizeof(curandStateXORWOW_t) * num_of_pixels));
    when("�����������ڴ�ķ���\n");

    // ������������ĳ�ʼ������
    // UPDATE ���������ʼ���� 1xnum_of_pixels ��Ϊ num_of_pixelsx1��ǰ�߻ᳬ���߳�������
    init_constant();
    init_random_library<<<num_of_pixels, 1>>>(states);

    // ����������ͳ����ĳ�ʼ��
    checkCudaErrors(cudaDeviceSynchronize());
    when("���������ͳ����ĳ�ʼ��\n");

    // UPDATE hittable_list ��Ҫ�� vector Ǩ�Ƶ����飬ʹ��ָ�뿪�ٿռ䣬�������Կ��䴫������
    // UPDATE hittable_list ������Ǩ�Ƶ� thrust_vector�����鲻���㴦��̳�����
    // UPDATE hittable_list ����ʹ����ָ��ʵ�֣��������Կ��ϴ���
    // �� cuda �ĺ����д���������������ΪҪʹ�� new ������������ʹ�� malloc ֱ�Ӵ���Ȼ�󿽱�
    hittable **dev_lists, **dev_world;
    camera **dev_camera;
    checkCudaErrors(cudaMalloc((void **) &dev_lists, sizeof(hittable *) * num_of_objects));
    checkCudaErrors(cudaMalloc((void **) &dev_world, sizeof(hittable *)));
    checkCudaErrors(cudaMalloc((void **) &dev_camera, sizeof(camera *)));
    when("������塢���硢������ڴ����\n");

    random_scene<<<num_of_objects, 1>>>(dev_lists, dev_world, dev_camera, image_width, image_height, states,
                                        num_of_objects);
    when("������塢���硢����Ĵ���\n");

    // ���䱾�غ��Կ�ͼ��Ŀռ�
    static color image[num_of_pixels];
    color *dev_image;
    checkCudaErrors(cudaMalloc((void **) &dev_image, sizeof(color) * num_of_pixels));
    when("���ͼ��ռ�ķ���\n");

    // ������硢�����ͼ���ڴ�ĳ�ʼ��
    checkCudaErrors(cudaDeviceSynchronize());
    when("��ʼ��Ⱦ\n");

    render<<<grids, threads>>>(samples_per_pixel, dev_camera, dev_world, max_depth, image_width, image_height,
                               dev_image, states);
    checkCudaErrors(cudaDeviceSynchronize());
    when("�����Ⱦ\n");

    // ���
    checkCudaErrors(cudaMemcpy(image, dev_image, sizeof(color) * num_of_pixels, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    when("���ͼ��Ŀ���\n");
    printf("P3\n%d %d\n255\n", image_width, image_height);

    for (int j = image_height - 1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {
            write_color(stdout, image[j * image_width + i], samples_per_pixel);
        }
    }

    FILE *fp = fopen("gpu-cuda-version-time.log", "a");
    fprintf(fp,
            "basic cuda versions, improve world generate, image width: %d,image height: %d, max depth: %d, samples per pixel: %d, time: %f s\n",
            image_width, image_height, max_depth, samples_per_pixel, (clock() - start) / float(CLOCKS_PER_SEC));
    fclose(fp);

    // �����˳�����
    free_scene<<<1, 1>>>(dev_lists, dev_world, dev_camera, num_of_objects);
    checkCudaErrors(cudaFree(dev_lists));
    checkCudaErrors(cudaFree(dev_world));
    checkCudaErrors(cudaFree(dev_camera));
    checkCudaErrors(cudaFree(dev_image));
    checkCudaErrors(cudaFree(states));
    cudaDeviceReset();
}