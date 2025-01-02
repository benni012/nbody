#include "body.h"
#include "nbody_cuda.cuh"
#include <cstddef>

#define BLOCK_SIZE 512
#define nStreams 1

// constant memory
__constant__ float G = 6.67408e-11;

int numBlocks;
int totalPairs;
int numPairBlocks;

int updateChunks;
int updateBlocks;

body_t *d_bodies;

cudaStream_t streams[nStreams];

__global__ void naive_kernel(body_t *bodies, int pointCount) {
  int particle_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (particle_idx >= pointCount)
    return;

  float4 particle = bodies[particle_idx].position; // Load into register

  __shared__ float4 shared_pos[BLOCK_SIZE];
  for (int i = 0; i < pointCount; i += BLOCK_SIZE) {
    int other_idx = i + threadIdx.x;

    if (other_idx >= pointCount)
      return;
    shared_pos[threadIdx.x] = bodies[other_idx].position;

    __syncthreads();

#pragma unroll
    for (int j = 0; j < BLOCK_SIZE; j++) {
      if (i + j >= pointCount) break;

      float4 other = shared_pos[j]; // Load into register
      float dx = other.x - particle.x;
      float dy = other.y - particle.y;
      float dz = other.z - particle.z;

      float distSq = fmaxf(dx * dx + dy * dy + dz * dz, 1e-4f); // Avoid small distances
      float invDist = rsqrtf(distSq);
      float invDist3 = invDist * invDist * invDist;
      float force = G * particle.w * other.w * invDist3;
      float fw = force/particle.w;
      bodies[particle_idx].velocity.x += dx * fw ;
      bodies[particle_idx].velocity.y += dy * fw ;
      bodies[particle_idx].velocity.z += dz * fw ;
    }
    __syncthreads();
  }
}

__global__ void update_kernel(body_t *bodies, int pointCount) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < pointCount) {
    atomicAdd(&bodies[i].position.x, bodies[i].velocity.x);
    atomicAdd(&bodies[i].position.y, bodies[i].velocity.y);
    atomicAdd(&bodies[i].position.z, bodies[i].velocity.z);
  }
  __syncthreads();
}

void pinMem(int N, body_t *bodies) {
  cudaMallocHost(&bodies, N * sizeof(body_t));
}

void setupGPU(body_t *bodies, int N) {
  numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  totalPairs = (N * (N - 1)) / 2;
  numPairBlocks = (totalPairs + BLOCK_SIZE - 1) / BLOCK_SIZE;

  updateChunks = (N + nStreams - 1) / nStreams;
  updateBlocks = (updateChunks + BLOCK_SIZE - 1) / BLOCK_SIZE;

  cudaMalloc(&d_bodies, N * sizeof(body_t));
  cudaMemcpy(d_bodies, bodies, N * sizeof(body_t), cudaMemcpyHostToDevice);

  // Create Streams
  for (int i = 0; i < nStreams; ++i) {
    cudaStreamCreate(&streams[i]);
  }
}

void gpu_update_naive(int N, body_t *bodies) {
  // Launch naive kernel
  naive_kernel<<<numPairBlocks, BLOCK_SIZE>>>(d_bodies, N);
  cudaDeviceSynchronize();
  cudaCheckErrors("STEP Kernel execution failed");

  // Launch update kernel for each stream
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * updateChunks;
    int currentChunkSize = std::min(updateChunks, N - offset);

    if (currentChunkSize > 0) {
      // Launch update kernel on the current stream
      update_kernel<<<numBlocks, BLOCK_SIZE, 0, streams[i]>>>(
          d_bodies + offset, currentChunkSize);
      cudaCheckErrors("UPDATE Kernel execution failed");

      cudaMemcpyAsync(&(bodies[offset]), d_bodies + offset,
                      currentChunkSize * sizeof(body_t),
                      cudaMemcpyDeviceToHost, streams[i]);
    }
  }

  // Synchronize all streams
  for (int i = 0; i < nStreams; ++i) {
    cudaStreamSynchronize(streams[i]);
  }
}

void cleanupGPU(body_t *bodies) {
  cudaFree(d_bodies);
  cudaFreeHost(bodies);

  // Destroy streams
  for (int i = 0; i < nStreams; ++i) {
    cudaStreamDestroy(streams[i]);
  }
}
