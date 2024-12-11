//
// Created by Ben Chwalek on 11.12.24.
//

#ifndef NBODY_OCTREE_H
#define NBODY_OCTREE_H
#define LEAF_CAPACITY 8
#define ROOT 0
#include <vector>

std::vector<int> octree;

typedef struct box {
    float3 center;
    float3 half_extent;
} box_t;

typedef struct node {
    int children;
    box_t box;
} node_t;

typedef struct data {
    float4 position[LEAF_CAPACITY];
    float3 velocity[LEAF_CAPACITY];
    int count;
} data_t;

typedef struct octree {
    std::vector<node_t> nodes;
    std::vector<data_t> data;
} octree_t;

void octree_init(octree_t *octree, float3 center, float3 half_extent) {
    octree->nodes.push_back({ROOT, {center, half_extent}});
    octree->data.push_back({});
}

void octree_split(octree_t *octree, int node) {
    node_t parent = octree->nodes[node];
    float3 half = parent.box.half_extent;
    half.x /= 2;
    half.y /= 2;
    half.z /= 2;

    octree->nodes[node].children = octree->nodes.size();
    for (int i = 0; i < 8; i++) {
        float3 center = parent.box.center;
        center.x += half.x * (i & 1 ? 1 : -1);
        center.y += half.y * (i & 2 ? 1 : -1);
        center.z += half.z * (i & 4 ? 1 : -1);

        octree->nodes.push_back({ROOT, {center, half}});
    }
}

void octree_insert(octree_t *octree, float4 position, float3 velocity) {
    int current = ROOT;
    while (octree->nodes[current].children != ROOT) {
        int child = 0;
        if (position.x > octree->nodes[current].box.center.x) child |= 1;
        if (position.y > octree->nodes[current].box.center.y) child |= 2;
        if (position.z > octree->nodes[current].box.center.z) child |= 4;
        current = octree->nodes[current].children + child;
    }
    int leaf = current;
    if (octree->data[leaf].count == LEAF_CAPACITY) {
        // split

    }
    octree->data[leaf].position[octree->data[leaf].count] = position;
}
#endif //NBODY_OCTREE_H
