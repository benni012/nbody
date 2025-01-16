#include "nbody_cuda.cuh"
#include "structures.h"
#include "timer.h"
#include <cmath>
#include <cstddef>
#include <iterator>
#include <thrust/partition.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#define BLOCK_SIZE 128
#define MAX_STACK_DEPTH 64
#define nStreams 12

// constant memory
__constant__ float G = 6.67408e-11;
__constant__ float theta_sq = 1.0f * 1.0f;
__constant__ float eps_sq = 0.05f * 0.05f;

int numBlocks;
int totalPairs;
int numPairBlocks;

int updateChunks;
int updateBlocks;

Timer tim;

body_t *d_bodies;

node_t *d_nodes;
octree_t *d_octree;
bool bh_setup = false;

cudaStream_t streams[nStreams];

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
          dx * dx + dy * dy + dz * dz + eps_sq; // Avoid small distances
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

__global__ void octree_init_kernel(octree_t *octree, float3 center, float half_extent,
                            int max_nodes) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    // first reset
    if (octree->nodes != nullptr) {
        delete[] octree->nodes;
        octree->nodes = nullptr;
    }
    octree->num_nodes = 0;
    octree->max_nodes = 0;

    // then init new
    octree->nodes = new node_t[max_nodes];
    octree->num_nodes = 1;
    octree->max_nodes = max_nodes;

    // init root node
    octree->nodes[ROOT] = {ROOT, {center, half_extent}, {0, 0, 0, 0}, 0, 0, 0};
  }
}

__global__ void build_octree_kernel(octree_t *octree, body_t *bodies, int N) {
    octree->nodes[ROOT].pos_idx = 0;
    octree->nodes[ROOT].count = N;

    int node = 0;
    while (node < octree->num_nodes) {
        if (octree->nodes[node].count > LEAF_CAPACITY) {
            octree_split_kernel<<<1,1>>>(octree, node, bodies);
        } else {
            for (int i = octree->nodes[node].pos_idx; i < octree->nodes[node].pos_idx + octree->nodes[node].count; i++) {
                body_t body = bodies[i];
                octree->nodes[node].center_of_mass.x += body.position.x * body.position.w;
                octree->nodes[node].center_of_mass.y += body.position.y * body.position.w;
                octree->nodes[node].center_of_mass.z += body.position.z * body.position.w;
                octree->nodes[node].center_of_mass.w += body.position.w;
            }
            if (octree->nodes[node].center_of_mass.w != 0) {
                octree->nodes[node].center_of_mass.x /= octree->nodes[node].center_of_mass.w;
                octree->nodes[node].center_of_mass.y /= octree->nodes[node].center_of_mass.w;
                octree->nodes[node].center_of_mass.z /= octree->nodes[node].center_of_mass.w;
            }
        }
        node++;
        __syncthreads();
    }
}

struct CompareZ {
    float z;
    __device__ CompareZ(float z_) : z(z_) {}
    __device__ bool operator()(const body_t& a) const {
        return a.position.z < z;
    }
};

struct CompareY {
    float y;
    __device__ CompareY(float y_) : y(y_) {}
    __device__ bool operator()(const body_t& a) const {
        return a.position.y < y;
    }
};

struct CompareX {
    float x;
    __device__ CompareX(float x_) : x(x_) {}
    __device__ bool operator()(const body_t& a) const {
        return a.position.x < x;
    }
};

__global__ void octree_split_kernel(octree_t *octree, int node,
                                    body_t *bodies) {
  
    node_t parent = octree->nodes[node];
    float3 center = parent.box.center;
    
    int split[] = {parent.pos_idx, 0, 0, 0, 0, 0, 0, 0, parent.pos_idx + parent.count};
    
    split[4] = thrust::partition(thrust::device, bodies + split[0], bodies + split[8], CompareZ(center.z)) - bodies;
    split[2] = thrust::partition(thrust::device, bodies + split[0], bodies + split[4], CompareY(center.y)) - bodies;
    split[6] = thrust::partition(thrust::device, bodies + split[4], bodies + split[8], CompareY(center.y)) - bodies;
    split[1] = thrust::partition(thrust::device, bodies + split[0], bodies + split[2], CompareX(center.x)) - bodies;
    split[3] = thrust::partition(thrust::device, bodies + split[2], bodies + split[4], CompareX(center.x)) - bodies;
    split[5] = thrust::partition(thrust::device, bodies + split[4], bodies + split[6], CompareX(center.x)) - bodies;
    split[7] = thrust::partition(thrust::device, bodies + split[6], bodies + split[8], CompareX(center.x)) - bodies;


    float half = parent.box.half_extent;
    
    int children = octree->num_nodes;
    octree->nodes[node].children = children;

    int nexts[8] = {children + 1,
                    children + 2,
                    children + 3,
                    children + 4,
                    children + 5,
                    children + 6,
                    children + 7,
                    parent.next};

    // Ensure that there is enough space in the flat array
    if (octree->num_nodes + 8 > octree->max_nodes) {
        // Handle overflow (this can be expanded or error handling could be added)
        // For simplicity, we are just returning.
        return;
    }

    for (int i = 0; i < 8; i++) {
        float3 new_center = parent.box.center;
        new_center.x += half / 2 * (i & 1 ? 1 : -1);
        new_center.y += half / 2 * (i & 2 ? 1 : -1);
        new_center.z += half / 2 * (i & 4 ? 1 : -1);

        octree->nodes[octree->num_nodes++] = {ROOT,
                                              {new_center, half / 2},
                                              {0, 0, 0, 0},
                                              split[i], split[i + 1] - split[i],
                                              nexts[i]};
    }
}

__global__ void octree_calculate_proxies_kernel(octree_t *octree, int node) {
    if (octree->nodes[node].children == ROOT) {
        return;
    }
    for (int i = 0; i < 8; i++) {
        octree_calculate_proxies_kernel<<<1,1>>>(octree, octree->nodes[node].children + i);
    }
    __syncthreads();

    float4 center_of_mass = {0, 0, 0, 0};
    for (int i = 0; i < 8; i++) {
        node_t *child = &octree->nodes[octree->nodes[node].children + i];
        if (child->center_of_mass.w == 0) {
            continue;
        }
        center_of_mass.x += child->center_of_mass.x * child->center_of_mass.w;
        center_of_mass.y += child->center_of_mass.y * child->center_of_mass.w;
        center_of_mass.z += child->center_of_mass.z * child->center_of_mass.w;
        center_of_mass.w += child->center_of_mass.w;
    }

    __syncthreads();
    if (center_of_mass.w == 0) {
        return;
    }

    center_of_mass.x /= center_of_mass.w;
    center_of_mass.y /= center_of_mass.w;
    center_of_mass.z /= center_of_mass.w;
    octree->nodes[node].center_of_mass = center_of_mass;
}

__global__ void bh_kernel(body_t *bodies, octree_t *octree) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  __syncthreads();

  int node = ROOT;
  float3 acceleration = {0, 0, 0};
  float4 position = bodies[tid].position;

  while (true) {
    node_t n = octree->nodes[node];
    float dx = n.center_of_mass.x - position.x;
    float dy = n.center_of_mass.y - position.y;
    float dz = n.center_of_mass.z - position.z;
    float d_sq = dx * dx + dy * dy + dz * dz + eps_sq;

    if (4 * n.box.half_extent * n.box.half_extent < theta_sq * d_sq) {
      float inv_d = rsqrtf(d_sq);
      float acc = G * n.center_of_mass.w * inv_d * inv_d * inv_d;

      acceleration.x += dx * acc;
      acceleration.y += dy * acc;
      acceleration.z += dz * acc;

      if (n.next == ROOT) {
        break;
      }
      node = n.next;
      __syncthreads();
    } else if (n.children == ROOT) {
      for (int j = n.pos_idx; j < n.pos_idx + n.count; j++) {
        float4 other = bodies[j].position;
        float dx = other.x - position.x;
        float dy = other.y - position.y;
        float dz = other.z - position.z;
        float d_sq = dx * dx + dy * dy + dz * dz + eps_sq;

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
      __syncthreads();
    } else {
      node = n.children;
      __syncthreads();
    }
  }

  atomicAdd(&bodies[tid].velocity.x, acceleration.x);
  atomicAdd(&bodies[tid].velocity.y, acceleration.y);
  atomicAdd(&bodies[tid].velocity.z, acceleration.z);
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

void gpu_update_position(int N, body_t *bodies) {
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

void gpu_setup_bh(body_t *bodies, octree_t *octree, int N) {
  cudaMalloc(&d_octree, sizeof(octree_t));
  cudaMalloc(&d_nodes, octree->max_nodes * sizeof(node_t));

  octree_t h_octree = *octree;
  h_octree.nodes = d_nodes; // Update to device pointer
  cudaMemcpy(d_octree, &h_octree, sizeof(octree_t), cudaMemcpyHostToDevice);
  cudaCheckErrors("SETUP execution failed");

   tim = Timer(true);
}

void gpu_update_bh(int N, body_t *bodies, octree_t *octree) {
  tim.start();
  // cudaMemcpy(d_nodes, octree->nodes, octree->max_nodes * sizeof(node_t),
  //            cudaMemcpyHostToDevice);
  // cudaCheckErrors("H2D execution failed");
  // printf("H2D %f\n\n", tim.elapsed());

  bh_kernel<<<numBlocks, BLOCK_SIZE>>>(d_bodies, d_octree);
  cudaDeviceSynchronize();
  cudaCheckErrors("STEP Kernel execution failed");
  printf("BH Kernel %f\n", tim.elapsed());
  tim.start();

  gpu_update_position(N, bodies);
  cudaCheckErrors("D2H execution failed");
  printf("D2H %f\n\n", tim.stop());
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


void gpu_build_octree(float3 center, float half_extent, int N){
  tim.start();
  octree_init_kernel<<<1,1>>>(d_octree, center, half_extent, N);
  cudaDeviceSynchronize();
  cudaCheckErrors("INIT Kernel execution failed");
  printf("INIT Kernel %f\n", tim.elapsed());
  tim.start();

  build_octree_kernel<<<1,1>>>(d_octree, d_bodies, N);
  cudaDeviceSynchronize();
  cudaCheckErrors("BUILD Kernel execution failed");
  printf("BUILD Kernel %f\n", tim.elapsed());
  tim.start();

  octree_calculate_proxies_kernel<<<1,1>>>(d_octree, ROOT);
  cudaDeviceSynchronize();
  cudaCheckErrors("PROXY Kernel execution failed");
  printf("PROXY Kernel %f\n", tim.elapsed());
}
