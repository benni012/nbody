#include "benchmark.h"
#include "nbody_cuda.cuh"
#include "structures.h"
#include <cmath>

#define BLOCK_SIZE 128

// constant memory
// mimic main.cpp defaults
__constant__ float G = 1;
__constant__ float dt = 0.0011;
__constant__ float theta_sq = 1.0f * 1.0f;
__constant__ float eps_sq = 0.05f * 0.05f;

int numBlocks;

body_t *d_bodies;

node_t *d_nodes;
octree_t *d_octree;
bool bh_setup = false;

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
            bodies[particle_idx].velocity.x += dx * fw * dt;
            bodies[particle_idx].velocity.y += dy * fw * dt;
            bodies[particle_idx].velocity.z += dz * fw * dt;
        }
        __syncthreads();
    }
}

__global__ void bh_kernel(body_t *bodies, octree_t *__restrict octree) {
    int particle_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // make room for 3 levels of the octree
    // level 1: 1
    // level 2: 8
    // level 3: 64
    // total 73
    __shared__ node_t shared_nodes[73];
    for (int i = particle_idx % BLOCK_SIZE; i < fminf(octree->num_nodes, 73);
         i += BLOCK_SIZE) {
        shared_nodes[i] = octree->nodes[i];
    }
    __syncthreads(); // Ensure the memory is copied before continuing

    if (particle_idx >= octree->max_nodes)
        return;

    int node = ROOT;
    float3 acceleration = {0, 0, 0};
    float4 position = bodies[particle_idx].position;

    node_t n;
    while (true) {
        if (node < 73) {
            n = shared_nodes[node];
        } else {
            n = octree->nodes[node];
        }

        float dx = n.center_of_mass.x - position.x;
        float dy = n.center_of_mass.y - position.y;
        float dz = n.center_of_mass.z - position.z;
        float d_sq = dx * dx + dy * dy + dz * dz + eps_sq;

        if (4 * n.box.half_extent * n.box.half_extent <
            theta_sq * d_sq) { // far enough for approx
            float inv_d = rsqrtf(d_sq);
            float acc = G * n.center_of_mass.w * inv_d * inv_d * inv_d;

            acceleration.x += dx * acc;
            acceleration.y += dy * acc;
            acceleration.z += dz * acc;

            if (n.next == ROOT || n.next >= octree->num_nodes) {
                break;
            }
            node = n.next;
        } else if (n.children == ROOT) { // Reached Leaf
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
            if (n.next == ROOT || n.next >= octree->num_nodes) {
                break;
            }
            node = n.next;
        } else { // Keep Iterating
            if (n.children >= octree->num_nodes) {
                break;
            }
            node = n.children;
        }
    }

    bodies[particle_idx].velocity.x += acceleration.x * dt;
    bodies[particle_idx].velocity.y += acceleration.y * dt;
    bodies[particle_idx].velocity.z += acceleration.z * dt;
}

__global__ void update_pos_kernel(int pointCount, body_t *bodies) {
    int particle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_idx < pointCount) {
        bodies[particle_idx].position.x += bodies[particle_idx].velocity.x * dt;
        bodies[particle_idx].position.y += bodies[particle_idx].velocity.y * dt;
        bodies[particle_idx].position.z += bodies[particle_idx].velocity.z * dt;
    }
    __syncthreads();
}

void gpu_update_position(int N, body_t *bodies) {
    BENCHMARK_START("PosUpdate_GPU");
    update_pos_kernel<<<numBlocks, BLOCK_SIZE>>>(N, d_bodies);
    cudaDeviceSynchronize();
    BENCHMARK_STOP("PosUpdate_GPU");
    cudaCheckErrors("UPDATE Kernel execution failed");

    BENCHMARK_START("BodiesD2H");
    cudaMemcpy(bodies, d_bodies, N * sizeof(body_t), cudaMemcpyDeviceToHost);
    BENCHMARK_STOP("BodiesD2H");
}

void gpu_pin_mem(int N, body_t *bodies) {
    cudaMallocHost(&bodies, N * sizeof(body_t)); // pin host mem
}

void gpu_setup(int N, body_t *bodies) {
    // kernel dims
    numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // allocate and cpy bodies to device
    cudaMalloc(&d_bodies, N * sizeof(body_t));
    cudaMemcpy(d_bodies, bodies, N * sizeof(body_t), cudaMemcpyHostToDevice);
}

void gpu_update_naive(int N, body_t *bodies) {
    BENCHMARK_START("AccUpdateNaive_GPU");
    naive_kernel<<<numBlocks, BLOCK_SIZE>>>(N, d_bodies);
    cudaDeviceSynchronize();
    BENCHMARK_STOP("AccUpdateNaive_GPU");
    cudaCheckErrors("STEP Kernel execution failed");
    gpu_update_position(N, bodies);
}

void gpu_cleanup_naive(body_t *bodies) {
    cudaFree(d_bodies);
    cudaFreeHost(bodies);
}

void gpu_setup_bh(body_t *bodies, octree_t *octree, int N) {
    cudaMalloc(&d_octree, sizeof(octree_t));
    cudaMalloc(&d_nodes, octree->max_nodes * sizeof(node_t));

    octree_t h_octree = *octree;
    h_octree.nodes = d_nodes; // Update to device pointer
    cudaMemcpy(d_octree, &h_octree, sizeof(octree_t), cudaMemcpyHostToDevice);
}

void gpu_update_bh(int N, body_t *bodies, octree_t *octree) {
    if (!bh_setup) {
        gpu_setup_bh(bodies, octree, N);
        bh_setup = true;
    }

    BENCHMARK_START("OctreeH2D");
    cudaMemcpy(d_nodes, octree->nodes, octree->max_nodes * sizeof(node_t),
               cudaMemcpyHostToDevice);
    BENCHMARK_STOP("OctreeH2D");

    BENCHMARK_START("AccUpdateBH_GPU");
    bh_kernel<<<numBlocks, BLOCK_SIZE>>>(d_bodies, d_octree);
    cudaDeviceSynchronize();
    BENCHMARK_STOP("AccUpdateBH_GPU");
    cudaCheckErrors("STEP Kernel execution failed");

    gpu_update_position(N, bodies);
}

void gpu_cleanup_bh(body_t *bodies) {
    cudaFree(d_bodies);
    cudaFreeHost(bodies);
    cudaFree(d_octree);
    cudaFree(d_nodes);
}
