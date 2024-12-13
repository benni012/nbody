//
// Created by Ben Chwalek on 11.12.24.
//

#ifndef NBODY_OCTREE_H
#define NBODY_OCTREE_H
#define LEAF_CAPACITY 8
#define ROOT 0
#include <vector>

typedef struct box {
    float3 center;
    float3 half_extent;
} box_t;

typedef struct node {
    int children;
    box_t box;
    float4 position;
    int next;
} node_t;

typedef struct octree {
    std::vector<node_t> nodes;
//    std::vector<data_t> data;
} octree_t;

void octree_init(octree_t *octree, float3 center, float3 half_extent) {
    octree->nodes = std::vector<node_t>();
    octree->nodes.push_back({ROOT, {center, half_extent}, {0, 0, 0, 0}, 0});
//    octree->data.push_back({{}, {}, 0});
}

void octree_split(octree_t *octree, int node) {
    node_t parent = octree->nodes[node];
    float3 half = parent.box.half_extent;
    half.x /= 2;
    half.y /= 2;
    half.z /= 2;

    int children = octree->nodes.size();
    octree->nodes[node].children = children;

    int nexts[8] = {children + 1,
                    children + 2,
                    children + 3,
                    children + 4,
                    children + 5,
                    children + 6,
                    children + 7,
                    parent.next};

    for (int i = 0; i < 8; i++) {
        float3 center = parent.box.center;
        center.x += half.x * (i & 1 ? 1 : -1);
        center.y += half.y * (i & 2 ? 1 : -1);
        center.z += half.z * (i & 4 ? 1 : -1);

        octree->nodes.push_back({ROOT,
                                 {center, half},
                                 {0, 0, 0, 0},
                                 nexts[i]});
    }
}

void octree_calculate_proxies(octree_t *octree, int node) {
    if (octree->nodes[node].children == ROOT) {
        return;
    }
    for (int i = 0; i < 8; i++) {
        octree_calculate_proxies(octree, octree->nodes[node].children + i);
    }
    float4 position = {0, 0, 0, 0};
    int count = 0;
    for (int i = 0; i < 8; i++) {
        if (octree->nodes[octree->nodes[node].children + i].position.w == 0) {
            continue;
        }
        position.x += octree->nodes[octree->nodes[node].children + i].position.x;
        position.y += octree->nodes[octree->nodes[node].children + i].position.y;
        position.z += octree->nodes[octree->nodes[node].children + i].position.z;
        position.w += octree->nodes[octree->nodes[node].children + i].position.w;
        count++;
    }
    if (count == 0) {
        return;
    }
    position.x /= count;
    position.y /= count;
    position.z /= count;
    octree->nodes[node].position = position;
}

void octree_insert(octree_t *octree, float4 position) {
    int current = ROOT;
    // check if within bounds
    if (position.x < octree->nodes[current].box.center.x - octree->nodes[current].box.half_extent.x ||
        position.x > octree->nodes[current].box.center.x + octree->nodes[current].box.half_extent.x ||
        position.y < octree->nodes[current].box.center.y - octree->nodes[current].box.half_extent.y ||
        position.y > octree->nodes[current].box.center.y + octree->nodes[current].box.half_extent.y ||
        position.z < octree->nodes[current].box.center.z - octree->nodes[current].box.half_extent.z ||
        position.z > octree->nodes[current].box.center.z + octree->nodes[current].box.half_extent.z) {
//        fprintf(stderr, "Position out of bounds\n");
        // print distance from center
//        fprintf(stderr, "Distance from center: %f %f %f\n", position.x - octree->nodes[current].box.center.x,
//                position.y - octree->nodes[current].box.center.y,
//                position.z - octree->nodes[current].box.center.z);
        return;
    }
    while (octree->nodes[current].children != ROOT) {
        int child = 0;
        if (position.x > octree->nodes[current].box.center.x) child |= 1;
        if (position.y > octree->nodes[current].box.center.y) child |= 2;
        if (position.z > octree->nodes[current].box.center.z) child |= 4;
        current = octree->nodes[current].children + child;
    }
    int leaf = current;
    if (octree->nodes[leaf].position.w == 0) {
        octree->nodes[leaf].position = position;
    } else if (position.x == octree->nodes[leaf].position.x &&
               position.y == octree->nodes[leaf].position.y &&
               position.z == octree->nodes[leaf].position.z) {
        octree->nodes[leaf].position.w += position.w;
    } else {
        octree_split(octree, leaf);
        float4 old_position = octree->nodes[leaf].position;
        octree_insert(octree, position);
        octree_insert(octree, old_position);
        octree->nodes[leaf].position = {0, 0, 0, 0};
    }
}

float3 octree_calculate_acceleration(octree_t *octree, float4 position, float theta) {
    if (position.w == 0) {
        return {0, 0, 0};
    }
    int node = ROOT;
    float3 acceleration = {0, 0, 0};
    do {
        float4 position2 = octree->nodes[node].position;
        if (position2.w == 0) {
            node = octree->nodes[node].next;
            continue;
        }
        float3 pos_to_pos2 = {position2.x - position.x,
                              position2.y - position.y,
                              position2.z - position.z};
        float dist = sqrtf(pos_to_pos2.x * pos_to_pos2.x +
                           pos_to_pos2.y * pos_to_pos2.y +
                           pos_to_pos2.z * pos_to_pos2.z);
        // if is leaf or approximation criterion is true, calculate acceleration
        if (octree->nodes[node].children == ROOT ||
            octree->nodes[node].box.half_extent.x < theta*dist) {

            float dist_cubed = dist*dist*dist;
            float acc = G * position2.w / (dist_cubed + 1e-4f);
            acceleration.x += pos_to_pos2.x * acc;
            acceleration.y += pos_to_pos2.y * acc;
            acceleration.z += pos_to_pos2.z * acc;
            // print acc


//            fprintf(stderr, "Acc: %f %f %f\n", acceleration.x, acceleration.y, acceleration.z);
            node = octree->nodes[node].next;
        } else {
            node = octree->nodes[node].children;
        }
    } while (node != ROOT);
    return acceleration;
}
#endif //NBODY_OCTREE_H
