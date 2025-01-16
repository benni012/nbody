#ifndef NBODY_NBODY_CUDA_H
#define NBODY_NBODY_CUDA_H

#include "structures.h"
#include <stdio.h>

// CUDA error-checking macro
#define cudaCheckErrors(msg)                                                   \
  do {                                                                         \
    cudaError_t __err = cudaGetLastError();                                    \
    if (__err != cudaSuccess) {                                                \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg,                  \
              cudaGetErrorString(__err), __FILE__, __LINE__);                  \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

__global__ void octree_init_kernel(octree_t *octree, float3 center, float half_extent,
                            int max_nodes);
__global__ void build_octree_kernel(octree_t *octree, body_t *bodies, int N);
__global__ void octree_split_kernel(octree_t *octree, int node, body_t *bodies);
__global__ void octree_calculate_proxies_kernel(octree_t *octree, int node);

void gpu_build_octree(float3 center, float half_extent, int N);
// Kernel declarations
__global__ void naive_kernel(int pointCount, body_t *bodies);
__global__ void bh_kernel(body_t *bodies, octree_t *octree);
__global__ void update_pos_kernel(int pointCount, body_t *bodies);

// Function declarations
void gpu_pin_mem(int N, body_t *bodies);
void gpu_setup(int N, body_t *bodies);
void gpu_update_postion(int N, body_t *bodies);

void gpu_update_naive(int N, body_t *bodies);
void gpu_cleanup_naive(body_t *bodies);

void gpu_setup_bh(body_t *bodies, octree_t *octree, int N);
void gpu_update_bh(int N, body_t *bodies, octree_t *octree);
void gpu_cleanup_bh(body_t *bodies);
#endif // NBODY_NBODY_CUDA_H
