#ifndef NBODY_STRUCTURES_H
#define NBODY_STRUCTURES_H

#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>

#ifdef CUDA_FOUND
#include <cuda.h>
#include <cuda_runtime.h>
#else
typedef struct float4 {
  float x, y, z, w;
} float4;
typedef struct float3 {
  float x, y, z;
} float3;

#endif 

#define ROOT 0
#define LEAF_CAPACITY 16

struct alignas(16) body_t {
    float4 position;
    float3 velocity;
};
struct alignas(16) box_t {
    float3 center;
    float half_extent;
};

struct alignas(32) node_t {
    int children;
    box_t box;
    float4 center_of_mass;
    int pos_idx;
    int count;
    int next;
};

struct alignas(32) octree_t {
    node_t* nodes;
    int num_nodes;
    int max_nodes;
};


#endif // NBODY_STRUCTURES_H
