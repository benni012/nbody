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
#include "nbody_cpu.h"
#include "graphics.h"

int main(int argc, char** argv) {
    // flags: --record --device=[cpu/gpu] --algo=[bh/naive] -n [number of particles]

    static struct option long_options[] = {
            {"record", no_argument, nullptr, 'r'},
            {"device", required_argument, nullptr, 'd'},
            {"algo", required_argument, nullptr, 'a'},
            {nullptr, 0, nullptr, 0}
    };

    int opt;
    int option_index = 0;
    bool record = false;
    bool use_gpu = false;
    bool use_bh = false;
    int N = 10e3;

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
                N = atoi(optarg);
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

    float4 positions[N];
    float3 velocities[N];

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
        double currentTime = glfwGetTime();

    if (use_gpu){
        gpu_update_naive(N, positions, velocities);
        
    } else {
        cpu_update_naive(N, positions, velocities);
        
    }
        // time it
        double frameTime = glfwGetTime() - currentTime;
        draw(positions, velocities, N, frameTime);
    }

// //    pclose(ffmpeg);

    cleanup(positions, velocities);
    return 0;
}
