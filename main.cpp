#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <getopt.h>

#define G 6.67408e-11
#define EPS_SQ (0.05 * 0.05)

#include "structures.h"

#ifdef CUDA_FOUND
#include "nbody_cuda.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include "octree.h"
#include "graphics.h"
#include "nbody_cpu.h"
octree_t octree;

int main(int argc, char **argv) {
  // flags: --record --device=[cpu/gpu] --algo=[bh/naive] -n [number of
  // particles]

  static struct option long_options[] = {
      {"record", no_argument, nullptr, 'r'},
      {"device", required_argument, nullptr, 'd'},
      {"algo", required_argument, nullptr, 'a'},
      {"num-particles", required_argument, nullptr, 'n'},
      {nullptr, 0, nullptr, 0}};

  int opt;
  int option_index = 0;
  bool record = false;
  bool use_gpu = false;
  bool use_bh = false;
  int N = 5000;

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

  body_t *bodies = (body_t *)calloc(N, sizeof(body_t));

  // Initialize everything
  initGraphics(N, bodies);
  { populate(bodies, N); }
#ifdef CUDA_FOUND
  gpu_pin_mem(N, bodies);
  gpu_setup(N, bodies);
  if (use_bh){
    // gpu_setup_bh(bodies, &octree, N);
  }
#endif

  //    // start ffmpeg telling it to expect raw rgba 720p-60hz frames
  //// -i - tells it to read frames from stdin
  //    const char* cmd = "ffmpeg -r 60 -f rawvideo -pix_fmt rgba -s 2560x1440
  //    -i - "
  //                      "-threads 0 -preset fast -y -pix_fmt yuv420p -crf 21
  //                      -vf vflip output.mp4";
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

    if (use_bh) {
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
      float3 center = {(min.x + max.x) / 2, 
                       (min.y + max.y) / 2,
                       (min.z + max.z) / 2};
      // create octree
      octree_init(&octree, center,
                  fmaxf(max.x - min.x, fmaxf(max.y - min.y, max.z - min.z)), N);
      octree_build(&octree, bodies, N);
      fprintf(stderr, "Time for init: %f\n", glfwGetTime() - current_time);
      current_time = glfwGetTime();

      octree_calculate_proxies(&octree, ROOT);
      fprintf(stderr, "Time for proxies: %f\n", glfwGetTime() - current_time);
      current_time = glfwGetTime();

      // gpu_build_octree(center, fmaxf(max.x - min.x, fmaxf(max.y - min.y, max.z - min.z)), N);

#ifdef CUDA_FOUND
      if (use_gpu) {
        gpu_update_bh(N, bodies, &octree);
      }
#endif
      if (!use_gpu && use_bh) {
      cpu_update_bh(N, bodies, &octree);
      }
      // fprintf(stderr, "Time for update: %f\n", glfwGetTime() - current_time);
    } else {
#ifdef CUDA_FOUND
      if (use_gpu) {
        gpu_update_naive(N, bodies);
      }
#endif
      if (!use_gpu && !use_bh) {
        cpu_update_naive(N, bodies);
      }
    }
    // time it
    float frame_time = glfwGetTime() - start_time;
    draw(bodies, N, frame_time);
  }

  //    pclose(ffmpeg);

  cleanup(bodies);
#ifdef CUDA_FOUND
  if (use_gpu) {
    gpu_cleanup_naive(bodies);
  }
#endif
  return 0;
}
