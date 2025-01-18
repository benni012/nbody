#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <getopt.h>

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

bool record = false;
bool use_gpu = false;
bool use_bh = false;
bool benchmark = false;
int N = 5000;

void signalHandler(int signum) {
  std::cout << "\nInterrupt signal (" << signum << ") received." << std::endl;
  Benchmark::getInstance().saveResults("nbody_benchmark.csv");
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
  exit(signum);
}

int main(int argc, char **argv) {
  // flags: --record --device=[cpu/gpu] --algo=[bh/naive] -n [number of
  // particles]

  static struct option long_options[] = {
      {"record", no_argument, nullptr, 'r'},
      {"device", required_argument, nullptr, 'd'},
      {"algo", required_argument, nullptr, 'a'},
      {"num-particles", required_argument, nullptr, 'n'},
      {"benchmark", no_argument, nullptr, 'b'},
      {nullptr, 0, nullptr, 0}};

  int opt;
  int option_index = 0;

  while ((opt = getopt_long(argc, argv, "rd:a:n:", long_options,
                            &option_index)) != -1) {
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
    case 'b':
      benchmark = true;
      break;
    case 'n':
      N = (int)strtod(optarg, nullptr);
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

  // Initialize everything
  init_graphics(N, bodies);
  { populate(bodies, N); }
#ifdef CUDA_FOUND
  gpu_pin_mem(N, bodies);
  gpu_setup(N, bodies);
#endif

  /*
        // start ffmpeg telling it to expect raw rgba 720p-60hz frames
        // -i - tells it to read frames from stdin
        const char* cmd = "ffmpeg -r 60 -f rawvideo -pix_fmt rgba -s 2560x1440-i
     - "
                          "-threads 0 -preset fast -y -pix_fmt yuv420p -crf 21
     -vf vflip output.mp4";

        // open pipe to ffmpeg's stdin in binary write mode
        FILE* ffmpeg = popen(cmd, "w");
        if (!ffmpeg) {
            fprintf(stderr, "Could not open pipe to ffmpeg\n");
            return -1;
        }
  */

  float zoom = 0.2;
  std::signal(SIGINT, signalHandler);

  while (!glfwWindowShouldClose(window)) {
    double start_time = glfwGetTime();
    glfwPollEvents();

    // scroll wheel or +/- keys
    if (glfwGetKey(window, GLFW_KEY_KP_ADD) == GLFW_PRESS ||
        glfwGetKey(window, GLFW_KEY_EQUAL) == GLFW_PRESS) {
      zoom *= 1.1; // Zoom in
    } else if (glfwGetKey(window, GLFW_KEY_KP_SUBTRACT) == GLFW_PRESS ||
               glfwGetKey(window, GLFW_KEY_MINUS) == GLFW_PRESS) {
      zoom /= 1.1; // Zoom out
    }

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
      octree_init(&octree, center,
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
        BENCHMARK_STOP("UpdateBH_CPU");
      }
    }
    // time it
    float frame_time = glfwGetTime() - start_time;
    draw(bodies, N, frame_time, zoom);
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
  Benchmark::getInstance().saveResults("benchmark_results.csv");
  return 0;
}
