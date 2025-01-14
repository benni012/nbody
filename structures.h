#ifndef NBODY_STRUCTURES_H
#define NBODY_STRUCTURES_H

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
typedef struct box {
    float3 center;
    float half_extent;
} box_t;

typedef struct node {
    int children;
    box_t box;
    float4 center_of_mass;
    int pos_idx;
    int count;
    int next;
} node_t;

typedef struct octree {
    node_t* nodes;
    int num_nodes;
    int max_nodes;
} octree_t;

#endif // NBODY_STRUCTURES_H
