#include "body.h"
#include "nbody_cuda.cuh"
#include <cstddef>

#define BLOCK_SIZE 1024
#define nStreams 4

// constant memory
__constant__ float G = 6.67408e-11;

int numBlocks;
int totalPairs;
int numPairBlocks;

int updateChunks;
int updateBlocks;

float4 *pinned_positions;
float4 *d_positions;
float3 *d_velocities;

cudaStream_t streams[nStreams];

__global__ void naive_kernel(float4 *positions, float3 *velocities,
                             int pointCount) {
  int particle_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (particle_idx >= pointCount)
    return;
  float4 particle_pos = positions[particle_idx]; // load into register

  __shared__ float4 shared_pos[BLOCK_SIZE];
  for (int i = 0; i < pointCount; i += BLOCK_SIZE) {
    int other_idx = i + threadIdx.x;
    if (other_idx >= pointCount)
      return;
    shared_pos[threadIdx.x] = positions[other_idx];

    __syncthreads();
#pragma unroll
    for (int j = 0; j < BLOCK_SIZE; j++) {
      if (i + j >= pointCount)
        break;
      float4 other_pos = shared_pos[j]; // load into register
      float dx = other_pos.x - particle_pos.x;
      float dy = other_pos.y - particle_pos.y;
      float dz = other_pos.z - particle_pos.z;
      float distSq =
          fmaxf(dx * dx + dy * dy + dz * dz, 1e-4f); // Avoid small distances
      float invDist = rsqrtf(distSq);
      float invDist3 = invDist * invDist * invDist;
      float force = G * particle_pos.w * other_pos.w * invDist3;
      float fw = force / particle_pos.w;
      velocities[particle_idx].x += dx * fw;
      velocities[particle_idx].y += dy * fw;
      velocities[particle_idx].z += dz * fw;
    }
    __syncthreads();
  }
}
__global__ void update_kernel(float4 *positions, float3 *velocities,
                              int pointCount) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < pointCount) {
    atomicAdd(&positions[i].x, velocities[i].x);
    atomicAdd(&positions[i].y, velocities[i].y);
    atomicAdd(&positions[i].z, velocities[i].z);
  }
}

void pinMem(int N, body_t *bodies) {
  cudaMallocHost(&bodies, N * sizeof(body_t));
  cudaMallocHost(&pinned_positions, N * sizeof(float4));
}

void setupGPU(body_t *bodies, int N) {
  numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  totalPairs = (N * (N - 1)) / 2;
  numPairBlocks = (totalPairs + BLOCK_SIZE - 1) / BLOCK_SIZE;

  updateChunks = (N + nStreams - 1) / nStreams;
  updateBlocks = (updateChunks + BLOCK_SIZE - 1) / BLOCK_SIZE;

  cudaMalloc(&d_positions, N * sizeof(float4));
  cudaMalloc(&d_velocities, N * sizeof(float3));

  // Copy positions and velocities from bodies to device
  float4 *positions = new float4[N];
  float3 *velocities = new float3[N];
  for (int i = 0; i < N; ++i) {
    positions[i] = bodies[i].position;
    velocities[i] = bodies[i].velocity;
  }

  cudaMemcpy(d_positions, positions, N * sizeof(float4),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_velocities, velocities, N * sizeof(float3),
             cudaMemcpyHostToDevice);

  delete[] positions;
  delete[] velocities;

  // Create Streams
  for (int i = 0; i < nStreams; ++i) {
    cudaStreamCreate(&streams[i]);
  }
}

void gpu_update_naive(int N, body_t *bodies) {
  // Launch naive kernel
  naive_kernel<<<numPairBlocks, BLOCK_SIZE>>>(d_positions, d_velocities, N);
  cudaDeviceSynchronize();
  cudaCheckErrors("STEP Kernel execution failed");

  // Launch update kernel for each stream
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * updateChunks;
    int currentChunkSize = std::min(updateChunks, N - offset);

    if (currentChunkSize > 0) {
      // Launch update kernel on the current stream
      update_kernel<<<numBlocks, BLOCK_SIZE, 1, streams[i]>>>(
          d_positions + offset, d_velocities + offset, currentChunkSize);
      cudaCheckErrors("UPDATE Kernel execution failed");

      cudaMemcpyAsync(pinned_positions, d_positions + offset,
                      currentChunkSize * sizeof(float4), cudaMemcpyDeviceToHost,
                      streams[i]);

      cudaStreamSynchronize(streams[i]); // Wait for the copy to complete
      for (int j = 0; j < currentChunkSize; ++j) {
        bodies[offset + j].position = pinned_positions[j];
      }
    }
  }

  // Synchronize all streams
  for (int i = 0; i < nStreams; ++i) {
    cudaStreamSynchronize(streams[i]);
  }
}
void cleanupGPU(body_t *bodies) {
  cudaFree(d_positions);
  cudaFree(d_velocities);
  cudaFreeHost(bodies);

  // Destroy streams
  for (int i = 0; i < nStreams; ++i) {
    cudaStreamDestroy(streams[i]);
  }
}
