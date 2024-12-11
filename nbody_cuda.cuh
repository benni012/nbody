#ifndef NBODY_NBODY_CUDA_H
#define NBODY_NBODY_CUDA_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256

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
void memoryMap(float4 *positions, float3 *velocities, int N);
void setupGPU(float4 *positions, float3 *velocities, int N);
void cleanup(float4 *positions, float3 *velocities);
void gpu_update_naive(int N, float4 *positions, float3 *velocities);

#endif // NBODY_NBODY_CUDA_H
