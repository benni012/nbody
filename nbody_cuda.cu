#include "nbody_cuda.cuh"
#define nStreams 4

// constant memory
__constant__ float G  = 6.67408e-11;


int numBlocks;
int totalPairs;
int numPairBlocks;

int updateChunks;
int updateBlocks;

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
      float distSq = fmaxf(dx * dx + dy * dy + dz * dz, 1e-4f); // Avoid small distances
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

void pinMem(float4 *positions, float3 *velocities, int N) {
  cudaMallocHost(&positions, N * sizeof(float4));
  cudaMallocHost(&velocities, N * sizeof(float3));
}

void setupGPU(float4 *positions, float3 *velocities, int N) {
  numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  totalPairs = (N * (N - 1)) / 2;
  numPairBlocks = (totalPairs + BLOCK_SIZE - 1) / BLOCK_SIZE;

  updateChunks = (N + nStreams - 1) / nStreams;
  updateBlocks = (updateChunks + BLOCK_SIZE - 1) / BLOCK_SIZE;

  cudaMalloc(&d_positions, N * sizeof(float4));
  cudaMalloc(&d_velocities, N * sizeof(float3));

  cudaMemcpy(d_positions, positions, N * sizeof(float4),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_velocities, velocities, N * sizeof(float3),
             cudaMemcpyHostToDevice);

  // Create Streams
  for (int i = 0; i < nStreams; ++i) {
    cudaStreamCreate(&streams[i]);
  }
}

void gpu_update_naive(int N, float4 *positions, float3 *velocities) {

  naive_kernel<<<numPairBlocks, BLOCK_SIZE>>>(d_positions, d_velocities, N);
  cudaDeviceSynchronize();
  cudaCheckErrors("STEP Kernel execution failed");

  for (int i = 0; i < nStreams; ++i) {
    int offset = i * updateChunks;
    int currentChunkSize = std::min(updateChunks, N - offset);

    if (currentChunkSize > 0) {
      // Launch update kernel on the current stream
      update_kernel<<<numBlocks, BLOCK_SIZE, 0, streams[i]>>>(
          d_positions + offset, d_velocities + offset, currentChunkSize);
      cudaCheckErrors("UPDATE Kernel execution failed");

      // // Perform asynchronous memory copy from device to host
      cudaMemcpyAsync(positions + offset, d_positions + offset,
                      currentChunkSize * sizeof(float4), cudaMemcpyDeviceToHost,
                      streams[i]);
    }
  }

  // Synchronize all streams
  for (int i = 0; i < nStreams; ++i) {
    cudaStreamSynchronize(streams[i]);
  }

  // update_kernel<<<numBlocks, BLOCK_SIZE>>>(d_positions, d_velocities, N);
  // cudaDeviceSynchronize();
  // cudaCheckErrors("UPDATE Kernel execution failed");
  // cudaMemcpy(positions, d_positions, N * sizeof(float4),
  // cudaMemcpyDeviceToHost);
}

void cleanup(float4 *positions, float3 *velocities) {
  cudaFree(d_positions);
  cudaFree(d_velocities);
  cudaFreeHost(positions);
  cudaFreeHost(velocities);

  // Destroy streams
  for (int i = 0; i < nStreams; ++i) {
    cudaStreamDestroy(streams[i]);
  }
}
