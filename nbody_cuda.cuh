#ifndef NBODY_NBODY_CUDA_H
#define NBODY_NBODY_CUDA_H

#include "body.h"
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

// Kernel declarations
__global__ void naive_kernel(float4 *positions, float3 *velocities, int pointCount);
__global__ void update_kernel(float4 *positions, float3 *velocities, int pointCount);

// Function declarations
void pinMem(int N, body_t *bodies);
void setupGPU(body_t *bodies, int N);
void cleanupGPU(body_t *bodies);
void gpu_update_naive(int N, body_t *bodies);

#endif // NBODY_NBODY_CUDA_H
