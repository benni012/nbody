#include "nbody_cuda.cuh"

float4 *d_positions;
float3 *d_velocities;

__global__ void naive_kernel(float4 *positions, float3 *velocities, int pointCount) {
  int N = pointCount;
  float G = 6.67408e-11; 
  int particle_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (particle_idx >= N) return;
  float4 particle_pos = positions[particle_idx];  // load into register

  __shared__ float4 shared_pos[BLOCK_SIZE];  
  for (int i = 0; i < N; i += BLOCK_SIZE) {
    int other_idx = i + threadIdx.x;
    if (other_idx >= N) return;
    shared_pos[threadIdx.x] = positions[other_idx];

    __syncthreads();
    #pragma unroll
    for (int j = 0; j < BLOCK_SIZE; j++){
      if (i+j >= N) break;
      float4 other_pos = shared_pos[j];  // load into register
      float dx = other_pos.x - particle_pos.x;
      float dy = other_pos.y - particle_pos.y;
      float dz = other_pos.z - particle_pos.z;
      float distSq = dx * dx + dy * dy + dz * dz;
      if (distSq < 1e-4f)
        distSq = 1e-4f;
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

__global__ void update_kernel(float4 *positions, float3 *velocities, int pointCount) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < pointCount) {
    positions[i].x += velocities[i].x;
    positions[i].y += velocities[i].y;
    positions[i].z += velocities[i].z;
  }
}

void memoryMap(float4 *positions, float3 *velocities, int N){
  cudaMallocHost(&positions, N * sizeof(float4));
  cudaMallocHost(&velocities, N * sizeof(float3));
}

void setupGPU(float4 *positions, float3 *velocities, int N) {
  cudaMalloc(&d_positions, N * sizeof(float4));
  cudaMalloc(&d_velocities, N * sizeof(float3));

  cudaMemcpy(d_positions, positions, N * sizeof(float4), cudaMemcpyHostToDevice);
  cudaMemcpy(d_velocities, velocities, N * sizeof(float3), cudaMemcpyHostToDevice);
}

void gpu_update_naive(int N, float4 *positions, float3 *velocities) {
  int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int totalPairs = (N * (N - 1)) / 2;
  int numPairBlocks = (totalPairs + BLOCK_SIZE - 1) / BLOCK_SIZE;

  naive_kernel<<<numPairBlocks, BLOCK_SIZE>>>(d_positions, d_velocities, N);
  cudaDeviceSynchronize();
  cudaCheckErrors("STEP Kernel execution failed");
  update_kernel<<<numBlocks, BLOCK_SIZE>>>(d_positions, d_velocities, N);
  cudaDeviceSynchronize();
  cudaCheckErrors("UPDATE Kernel execution failed");
  cudaMemcpy(positions, d_positions, N * sizeof(float4), cudaMemcpyDeviceToHost);
}

void cleanup(float4 *positions, float3 *velocities){
  cudaFree(d_positions);
  cudaFree(d_velocities);
  cudaFreeHost(positions);
  cudaFreeHost(velocities);
 
}
