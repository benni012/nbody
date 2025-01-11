#ifndef NBODY_STRUCTURES_H
#define NBODY_STRUCTURES_H

#ifdef CUDA_FOUND
#include <cuda.h>
#include <cuda_runtime.h>
#endif 


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
    node_t* nodes;  // Flat array of nodes
    int num_nodes;
    int max_nodes;  // Maximum number of nodes in the flat array
} octree_t;

#endif // NBODY_STRUCTURES_H
