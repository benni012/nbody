#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cmath>
#include <random>
#include <chrono>
#include <getopt.h>
#include <cstring>

#define G 6.67408e-11

#ifdef CUDA_FOUND
#include "nbody_cuda.cuh"
#else
typedef struct float4 {
    float x, y, z, w;
} float4;

typedef struct float3 {
    float x, y, z;
} float3;
#endif
#include "graphics.h"
#include "nbody_cpu.h"
#include "octree.h"
octree_t octree;

int main(int argc, char** argv) {
    // flags: --record --device=[cpu/gpu] --algo=[bh/naive] -n [number of particles]

    static struct option long_options[] = {
            {"record", no_argument, nullptr, 'r'},
            {"device", required_argument, nullptr, 'd'},
            {"algo", required_argument, nullptr, 'a'},
            {"num-particles", required_argument, nullptr, 'n'},
            {nullptr, 0, nullptr, 0}
    };

    int opt;
    int option_index = 0;
    bool record = false;
    bool use_gpu = false;
    bool use_bh = false;
    int N = 5000;

    while ((opt = getopt_long(argc, argv, "rd:a:n:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'r':
                record = true;
                break;
            case 'd':
                if (strcmp(optarg, "gpu") == 0) {
                    use_gpu = true;
                } else if (strcmp(optarg, "cpu") == 0) {
                    use_gpu = false;
                } else {
                    fprintf(stderr, "Invalid device option\n");
                    return -1;
                }
                break;
            case 'a':
                if (strcmp(optarg, "bh") == 0) {
                    use_bh = true;
                } else if (strcmp(optarg, "naive") == 0) {
                    use_bh = false;
                } else {
                    fprintf(stderr, "Invalid algorithm option\n");
                    return -1;
                }
                break;
            case 'n':
                N = (int)strtod(optarg, nullptr);
                break;
            default:
                break;
        }
    }

#ifndef CUDA_FOUND
    if (use_gpu) {
        fprintf(stderr, "CUDA not found, falling back to CPU\n");
        use_gpu = false;
    }
#endif

    float4 *positions = (float4*)calloc(N, sizeof(float4));
    float3 *velocities = (float3*)calloc(N, sizeof(float3));
//    float3 velocities[N];

    // Initialize everything
    initGraphics(N, positions);
    if (use_gpu){
        populate(positions, velocities, N);
        memoryMap(positions, velocities, N);
        setupGPU(positions, velocities, N);
    } else {
        populate(positions, velocities, N);
    }

//    // start ffmpeg telling it to expect raw rgba 720p-60hz frames
//// -i - tells it to read frames from stdin
//    const char* cmd = "ffmpeg -r 60 -f rawvideo -pix_fmt rgba -s 2560x1440 -i - "
//                      "-threads 0 -preset fast -y -pix_fmt yuv420p -crf 21 -vf vflip output.mp4";
////    const char* cmd = "cat";
//
//// open pipe to ffmpeg's stdin in binary write mode
//    FILE* ffmpeg = popen(cmd, "w");
//    if (!ffmpeg) {
//        fprintf(stderr, "Could not open pipe to ffmpeg\n");
//        return -1;
//    }


    while (!glfwWindowShouldClose(window)) {
        double start_time = glfwGetTime();
//        glfwPollEvents();

        if (use_gpu) {
            gpu_update_naive(N, positions, velocities);
        } else if (use_bh) {
            double current_time = start_time;
            // find min/max of positions
            float3 min = {INFINITY, INFINITY, INFINITY};
            float3 max = {-INFINITY, -INFINITY, -INFINITY};
            for (int i = 0; i < N; i++) {
                min.x = fminf(min.x, positions[i].x);
                min.y = fminf(min.y, positions[i].y);
                min.z = fminf(min.z, positions[i].z);
                max.x = fmaxf(max.x, positions[i].x);
                max.y = fmaxf(max.y, positions[i].y);
                max.z = fmaxf(max.z, positions[i].z);
            }
            float3 center = {
                    (min.x + max.x) / 2,
                    (min.y + max.y) / 2,
                    (min.z + max.z) / 2
            };
            // create octree
            octree_init(&octree, center, fmaxf(max.x - min.x, fmaxf(max.y - min.y, max.z - min.z)));
//            octree_init(&octree, {0, 0, 0}, 1);
            auto cube = octree.nodes[ROOT].box;
//            float m = static_cast<float>(1 << 16);
//
//            std::sort(positions, positions + N, [&](const float4& a, const float4& b) {
//                auto z_index = [&](const float4& pos) {
//                    float normX = (cube.center.x - pos.x) / cube.half_extent + 0.5f;
//                    float normY = (cube.center.y - pos.y) / cube.half_extent + 0.5f;
//                    float normZ = (cube.center.z - pos.z) / cube.half_extent + 0.5f;
//
//                    uint16_t x = static_cast<uint16_t>(normX * m);
//                    uint16_t y = static_cast<uint16_t>(normY * m);
//                    uint16_t z = static_cast<uint16_t>(normZ * m);
//
//                    return ZOrder::index_of(x, y, z);
//                };
//
//                return z_index(a) < z_index(b);
//            });

            // shuffle both positions and velocities (the same though)
//            std::shuffle(positions, positions + N, std::default_random_engine(42));
//            std::shuffle(velocities, velocities + N, std::default_random_engine(42));


            for (int i = 0; i < N; i++) {
                octree_insert(&octree, positions[i]);
            }
            fprintf(stderr, "Time for init: %f\n", glfwGetTime() - current_time);
            current_time = glfwGetTime();

            octree_calculate_proxies(&octree, ROOT);
            fprintf(stderr, "Time for proxies: %f\n", glfwGetTime() - current_time);
            current_time = glfwGetTime();
            cpu_update_bh(N, positions, velocities, &octree);
            fprintf(stderr, "Time for update: %f\n", glfwGetTime() - current_time);
        } else {
            cpu_update_naive(N, positions, velocities);
        }

        // time it
        float frame_time = glfwGetTime() - start_time;
        draw(positions, velocities, N, frame_time);
    }

//    pclose(ffmpeg);

    cleanup(positions, velocities);
    return 0;
}
