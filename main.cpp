#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <getopt.h>
#include <sstream>
#include <string>

#define G 1
#define DT 0.001
#define EPS_SQ (0.05 * 0.05)

#ifdef CUDA_FOUND
#include "nbody_cuda.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include "benchmark.h"
#include "graphics.h"
#include "nbody_cpu.h"
#include "octree.h"
#include "structures.h"

octree_t octree;
body_t *bodies;

std::atomic<bool> close_window(false);

bool use_gpu = false;
bool use_bh = false;
bool benchmark = false;
int population_method = 3; // Plummers
int N = 5000;
unsigned int iters = UINT_MAX;

void signalHandler(int signum) {
    std::cout << "\nInterrupt signal (" << signum << ") received." << std::endl;
    Benchmark::getInstance().saveResults("nbody_benchmark.csv");
    close_window = true;

#ifdef CUDA_FOUND
    if (use_gpu) {
        if (use_bh) {
            gpu_cleanup_bh(bodies);
        } else {
            gpu_cleanup_naive(bodies);
        }
    }
#endif
    exit(signum);
}

int main(int argc, char **argv) {
    // flags:
    // --benchmark
    // --device=[cpu/gpu]
    // --algo=[bh/naive]
    // --num-particles [number of particles]
    // --population [population method]

    static struct option long_options[] = {
        {"benchmark", no_argument, nullptr, 'b'},
        {"device", required_argument, nullptr, 'd'},
        {"algo", required_argument, nullptr, 'a'},
        {"num-particles", required_argument, nullptr, 'n'},
        {"iters", required_argument, nullptr, 'i'},
        {"pop-method", required_argument, nullptr, 'p'},
        {nullptr, 0, nullptr, 0}};

    int opt;
    int option_index = 0;

    while ((opt = getopt_long(argc, argv, "bd:a:n:i:p:", long_options,
                              &option_index)) != -1) {
        switch (opt) {
        case 'b':
            benchmark = true;
            break;
        case 'd':
            if (strcmp(optarg, "gpu") == 0) {
                use_gpu = true;
            } else if (strcmp(optarg, "cpu") == 0) {
                use_gpu = false;
            } else {
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
        case 'i':
            iters = (int)strtod(optarg, nullptr);
            break;
        case 'p':
            population_method = (int)strtod(optarg, nullptr);
            break;
        default:
            break;
        }
    }

    Benchmark::getInstance().enableBenchmarking(benchmark);
#ifndef CUDA_FOUND
    if (use_gpu) {
        fprintf(stderr, "CUDA not found, falling back to CPU\n");
        use_gpu = false;
    }
#endif

    bodies = (body_t *)calloc(N, sizeof(body_t));

    // set default color
    for (int i = 0; i < N; i++) {
        bodies[i].color = {0.5, 0.1, 0.05, 1.0};
    }

    // Initialize everything
    init_graphics(N, bodies);
    float zoom = populate(bodies, N, population_method);
    if (zoom < 0) {
        fprintf(stderr, "Invalid population method\n");
        return -1;
    }
#ifdef CUDA_FOUND
    gpu_pin_mem(N, bodies);
    gpu_setup(N, bodies);
    // if (use_bh) {
        // octree_init(&octree, {0, 0, 0}, 1.0, N);
        // gpu_setup_bh(N, bodies, &octree);
    // }
#endif

    float phi = 1.570796f;       // Vertical angle (latitude)
    float theta = 1.570796f;       // Vertical angle (latitude)
    float sensitivity = 0.02f;

    std::signal(SIGINT, signalHandler);
    int current_iter = 0;
    while (!glfwWindowShouldClose(window) && !close_window) {
        double start_time = glfwGetTime();
        glfwPollEvents();

        // scroll wheel or +/- keys or arrow up/down (93/45 for german layout)
        if (glfwGetKey(window, GLFW_KEY_KP_ADD) == GLFW_PRESS ||
            glfwGetKey(window, GLFW_KEY_EQUAL) == GLFW_PRESS ||
            glfwGetKey(window, 93)) {
            zoom *= 1.1; // Zoom in
        }
        if (glfwGetKey(window, GLFW_KEY_KP_SUBTRACT) == GLFW_PRESS ||
                   glfwGetKey(window, GLFW_KEY_MINUS) == GLFW_PRESS ||
                   glfwGetKey(window, 47)) {
            zoom /= 1.1; // Zoom out
        }
        
        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)  theta -= sensitivity;  // Rotate left
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) theta += sensitivity;  // Rotate right
        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)    phi -= sensitivity;    // Rotate up
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)  phi += sensitivity;    // Rotate down

        theta = fmod(theta, 2.0f * M_PI);
        if (theta < 0) theta += 2.0f * M_PI;
        phi = std::max(0.1f, std::min(phi, (float)M_PI - 0.1f));

        if (use_bh) { // BARNES HUT BODY UPDATES
            octree_free(&octree);
            double current_time = start_time;
            // find min/max of positions
            float3 min = {INFINITY, INFINITY, INFINITY};
            float3 max = {-INFINITY, -INFINITY, -INFINITY};
            for (int i = 0; i < N; i++) {
                min.x = fminf(min.x, bodies[i].position.x);
                min.y = fminf(min.y, bodies[i].position.y);
                min.z = fminf(min.z, bodies[i].position.z);
                max.x = fmaxf(max.x, bodies[i].position.x);
                max.y = fmaxf(max.y, bodies[i].position.y);
                max.z = fmaxf(max.z, bodies[i].position.z);
            }
            float3 center = {(min.x + max.x) / 2, (min.y + max.y) / 2,
                             (min.z + max.z) / 2};
            // create octree
            BENCHMARK_START("OctreeInit");
            octree_init(
                &octree, center,
                fmaxf(max.x - min.x, fmaxf(max.y - min.y, max.z - min.z)), N);
            BENCHMARK_STOP("OctreeInit");

            BENCHMARK_START("OctreeBuild");
            octree_build(&octree, bodies, N);
            BENCHMARK_STOP("OctreeBuild");

            BENCHMARK_START("OctreeProxies");
            octree_calculate_proxies(&octree, ROOT);
            BENCHMARK_STOP("OctreeProxies");

#ifdef CUDA_FOUND
            if (use_gpu) {
                gpu_update_bh(N, bodies, &octree);
            }
#endif
            if (!use_gpu) {
                BENCHMARK_START("UpdateBH_CPU");
                cpu_update_bh(N, bodies, &octree);
                BENCHMARK_STOP("UpdateBH_CPU");
            }
        } else { // NAIVE BODY UPDATES
#ifdef CUDA_FOUND
            if (use_gpu) {
                gpu_update_naive(N, bodies);
            }
#endif
            if (!use_gpu) {
                BENCHMARK_START("UpdateNaive_CPU");
                cpu_update_naive(N, bodies);
                BENCHMARK_STOP("UpdateNaive_CPU");
            }
        }
        // time it
        float frame_time = glfwGetTime() - start_time;
        draw(bodies, N, frame_time, zoom, phi, theta);

        // needed because we catch ctrl-c signal
        if (close_window || current_iter >= iters - 1) { 
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            break;
        }
        current_iter++;
    }

    // pclose(ffmpeg);

    // CLEANUP
    cleanup_graphics(bodies);
#ifdef CUDA_FOUND
    if (use_gpu) {
        if (use_bh) {
            gpu_cleanup_bh(bodies);
        } else {
            gpu_cleanup_naive(bodies);
        }
    }
#endif

    std::ostringstream filename;
    filename << "benchmarks/benchmark_results_N" << N << "_BH" << (int)use_bh
             << "_GPU" << (int)use_gpu << ".csv";
    Benchmark::getInstance().saveResults(filename.str());
    return 0;
}
