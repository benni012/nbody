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
    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int warp_id = thread_id / warpSize;
    const int lane_id = thread_id % warpSize;

    // Initialize the root node (single thread for initialization)
    if (thread_id == 0) {
        octree->nodes[ROOT].pos_idx = 0;
        octree->nodes[ROOT].count = N;
    }
    __syncthreads();

    for (int node = 0; node < octree->num_nodes; node += blockDim.x * gridDim.x) {
        int current_node = node + thread_id;

        if (current_node < octree->num_nodes) {
            // Check if splitting is needed
            if (octree->nodes[current_node].count > LEAF_CAPACITY) {
                // Perform splitting
                octree_split_kernel<<<1, 1>>>(octree, current_node, bodies);
            } else {
                // Compute center of mass using warp-level reduction
                float4 com = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                int start = octree->nodes[current_node].pos_idx;
                int count = octree->nodes[current_node].count;

                for (int i = lane_id; i < count; i += warpSize) {
                    body_t body = bodies[start + i];
                    com.x = body.position.x * body.position.w;
                    com.y = body.position.y * body.position.w;
                    com.z = body.position.z * body.position.w;
                    com.w = body.position.w;
                }

                // Finalize the center of mass calculation
                if (lane_id == 0 && com.w > 0) {
                    com.x /= com.w;
                    com.y /= com.w;
                    com.z /= com.w;
                }

                // Assign center of mass to the node
                if (lane_id == 0) {
                    octree->nodes[current_node].center_of_mass = com;
                }
            }
        }
    }
}

struct CompareXYZ {
    float3 center;
    int octant;

    __device__ CompareXYZ(float3 center_, int octant_)
        : center(center_), octant(octant_) {}

    __device__ bool operator()(const body_t &body) const {
        bool x_cmp = (octant & 1) ? (body.position.x >= center.x)
                                  : (body.position.x < center.x);
        bool y_cmp = (octant & 2) ? (body.position.y >= center.y)
                                  : (body.position.y < center.y);
        bool z_cmp = (octant & 4) ? (body.position.z >= center.z)
                                  : (body.position.z < center.z);

        return x_cmp && y_cmp && z_cmp;
    }
};


__global__ void octree_split_kernel(octree_t *octree, int node, body_t *bodies) {
    // Each thread handles one octant
    int thread_id = threadIdx.x;

    // Parent node data
    node_t parent = octree->nodes[node];
    float3 center = parent.box.center;
    float half_extent = parent.box.half_extent / 2.0f;

    __shared__ int split_indices[9];  // Shared memory for split boundaries
    if (thread_id == 0) {
        split_indices[0] = parent.pos_idx;  // Start index of parent
        split_indices[8] = parent.pos_idx + parent.count;  // End index of parent
    }
    __syncthreads();

    if (thread_id < 8) {
        // Partition bodies into octants using CompareXYZ
        split_indices[thread_id + 1] = thrust::partition(
            thrust::device,
            bodies + split_indices[thread_id],      // Start of range
            bodies + split_indices[8],             // End of range
            CompareXYZ(center, thread_id)) - bodies;
    }
    __syncthreads();

    // Single thread creates child nodes
    if (thread_id == 0) {
        int children = octree->num_nodes;
        octree->nodes[node].children = children;

        if (octree->num_nodes + 8 > octree->max_nodes) {
            // Handle overflow (e.g., expand the octree or throw an error)
            return;
        }

        for (int i = 0; i < 8; i++) {
            // Calculate the new center for the child octant
            float3 new_center = {
                center.x + half_extent * ((i & 1) ? 1 : -1),
                center.y + half_extent * ((i & 2) ? 1 : -1),
                center.z + half_extent * ((i & 4) ? 1 : -1)
            };

            // Create child node
            octree->nodes[children + i] = {
                .children = ROOT,  // Leaf by default
                .box = {new_center, half_extent},
                .center_of_mass = {0, 0, 0, 0},
                .pos_idx = split_indices[i],
                .count = split_indices[i + 1] - split_indices[i],
                .next = (i == 7) ? parent.next : children + i + 1
            };
        }

        // Update octree node count
        octree->num_nodes += 8;
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
