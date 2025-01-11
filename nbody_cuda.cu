#include "structures.h"
#include "nbody_cuda.cuh"
#include <cstddef>
#include <iterator>
#include "timer.h"

#define BLOCK_SIZE 256
#define nStreams 64

// constant memory
__constant__ float G = 6.67408e-11;
__constant__ float theta_sq = 0.8f * 0.8f;

int numBlocks;
int totalPairs;
int numPairBlocks;

int updateChunks;
int updateBlocks;

body_t *d_bodies;

node_t *d_nodes;
octree_t *d_octree;
bool bh_setup = false;

cudaStream_t streams[nStreams];
size_t sharedMemSize = BLOCK_SIZE * sizeof(float4); // Shared memory for positions

__global__ void naive_kernel(int pointCount, body_t *bodies) {
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
      if (i + j >= pointCount)
        break;

      float4 other = shared_pos[j]; // Load into register
      float dx = other.x - particle.x;
      float dy = other.y - particle.y;
      float dz = other.z - particle.z;

      float distSq =
          fmaxf(dx * dx + dy * dy + dz * dz, 1e-4f); // Avoid small distances
      float invDist = rsqrtf(distSq);
      float invDist3 = invDist * invDist * invDist;
      float force = G * particle.w * other.w * invDist3;
      float fw = force / particle.w;
      bodies[particle_idx].velocity.x += dx * fw;
      bodies[particle_idx].velocity.y += dy * fw;
      bodies[particle_idx].velocity.z += dz * fw;
    }
    __syncthreads();
  }
}
__global__ void bh_kernel(int N, body_t *bodies, octree_t *octree) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = tid; i < N; i += stride) {
    int node = ROOT;
    float3 acceleration = {0, 0, 0};
    float4 position = bodies[i].position;

    while (true) {
      node_t n = octree->nodes[node];
      float dx = n.center_of_mass.x - position.x;
      float dy = n.center_of_mass.y - position.y;
      float dz = n.center_of_mass.z - position.z;
      float d_sq = fmaxf(dx * dx + dy * dy + dz * dz, 1e-4);

      if (n.box.half_extent * n.box.half_extent < theta_sq * d_sq) {
        float inv_d = rsqrtf(d_sq);
        float acc = G * n.center_of_mass.w * inv_d * inv_d * inv_d;

        acceleration.x += dx * acc;
        acceleration.y += dy * acc;
        acceleration.z += dz * acc;

        if (n.next == ROOT) {
          break;
        }
        node = n.next;
      } else if (n.children == ROOT) {
        for (int j = n.pos_idx; j < n.pos_idx + n.count; j++) {
          float4 other = bodies[j].position;
          float dx = other.x - position.x;
          float dy = other.y - position.y;
          float dz = other.z - position.z;
          float d_sq = fmaxf(dx * dx + dy * dy + dz * dz, 1e-4f);

          float inv_d = rsqrtf(d_sq);
          float acc = G * other.w * inv_d * inv_d * inv_d;

          acceleration.x += dx * acc;
          acceleration.y += dy * acc;
          acceleration.z += dz * acc;
        }
        if (n.next == ROOT) {
          break;
        }
        node = n.next;
      } else {
        node = n.children;
      }
    }

    atomicAdd(&bodies[i].velocity.x, acceleration.x);
    atomicAdd(&bodies[i].velocity.y, acceleration.y);
    atomicAdd(&bodies[i].velocity.z, acceleration.z);
  }
}

__global__ void update_pos_kernel(int pointCount, body_t *bodies) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < pointCount) {
    atomicAdd(&bodies[i].position.x, bodies[i].velocity.x);
    atomicAdd(&bodies[i].position.y, bodies[i].velocity.y);
    atomicAdd(&bodies[i].position.z, bodies[i].velocity.z);
  }
  __syncthreads();
}

void gpu_update_position(int N, body_t *bodies){
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * updateChunks;
    int currentChunkSize = std::min(updateChunks, N - offset);

    if (currentChunkSize > 0) {
      update_pos_kernel<<<numBlocks, BLOCK_SIZE, 0, streams[i]>>>(
          currentChunkSize, d_bodies + offset);
      cudaCheckErrors("UPDATE Kernel execution failed");

      cudaMemcpyAsync(&(bodies[offset]), d_bodies + offset,
                      currentChunkSize * sizeof(body_t), cudaMemcpyDeviceToHost,
                      streams[i]);
    }
  }

  for (int i = 0; i < nStreams; ++i) {
    cudaStreamSynchronize(streams[i]);
  }
}

void gpu_pin_mem(int N, body_t *bodies) {
  cudaMallocHost(&bodies, N * sizeof(body_t)); // pin host mem
}

void gpu_setup(int N, body_t *bodies) {
  // kernel dims
  numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  totalPairs = (N * (N - 1)) / 2;
  numPairBlocks = (totalPairs + BLOCK_SIZE - 1) / BLOCK_SIZE;

  updateChunks = (N + nStreams - 1) / nStreams;
  updateBlocks = (updateChunks + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // allocate and cpy bodies to device
  cudaMalloc(&d_bodies, N * sizeof(body_t));
  cudaMemcpy(d_bodies, bodies, N * sizeof(body_t), cudaMemcpyHostToDevice);

  // create copy device to host streams
  for (int i = 0; i < nStreams; ++i) {
    cudaStreamCreate(&streams[i]);
  }
}

void gpu_update_naive(int N, body_t *bodies) {
  naive_kernel<<<numPairBlocks, BLOCK_SIZE>>>(N, d_bodies);
  cudaDeviceSynchronize();
  cudaCheckErrors("STEP Kernel execution failed");
  gpu_update_position(N, bodies);
}

void gpu_cleanup_naive(body_t *bodies) {
  cudaFree(d_bodies);
  cudaFreeHost(bodies);

  // destroy streams
  for (int i = 0; i < nStreams; ++i) {
    cudaStreamDestroy(streams[i]);
  }
}

void gpu_setup_bh(body_t *bodies, octree_t *octree){
  cudaMalloc(&d_octree, sizeof(octree_t));
  cudaMalloc(&d_nodes, octree->max_nodes* sizeof(node_t));
  
  octree_t h_octree = *octree;
  h_octree.nodes = d_nodes; // Update to device pointer
  cudaMemcpy(d_octree, 
             &h_octree, 
             sizeof(octree_t),
             cudaMemcpyHostToDevice);
}


void gpu_update_bh(int N, body_t *bodies, octree_t *octree) {
  if (!bh_setup){
    gpu_setup_bh(bodies, octree);
    bh_setup = true;
  }
  
  // Timer tim = Timer(true);
  cudaMemcpy(d_nodes, octree->nodes, octree->max_nodes* sizeof(node_t),
             cudaMemcpyHostToDevice);
  // printf("H2D %f\n", tim.elapsed());
  // tim.start();
  
  bh_kernel<<<numBlocks, BLOCK_SIZE>>>(N, d_bodies, d_octree);
  cudaDeviceSynchronize();
  cudaCheckErrors("STEP Kernel execution failed");
  // printf("KER %f\n", tim.elapsed());
  // tim.start();

  update_pos_kernel<<<numBlocks, BLOCK_SIZE>>>(N, d_bodies);
  // printf("KEU %f\n", tim.elapsed());
  // tim.start();
  cudaMemcpy(bodies, d_bodies, N * sizeof(body_t), cudaMemcpyDeviceToHost);
  // gpu_update_poH2DH2Dsition(N, bodies);
  // printf("D2H %f\n\n", tim.stop());
}

void gpu_cleanup_bh() {
  cudaFree(d_bodies);
  cudaFree(d_octree);
  cudaFree(d_nodes);

  // Destroy streams
  for (int i = 0; i < nStreams; ++i) {
    cudaStreamDestroy(streams[i]);
  }
}
