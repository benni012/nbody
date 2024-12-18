//
// Created by Ben Chwalek on 11.12.24.
//

#ifndef NBODY_OCTREE_H
#define NBODY_OCTREE_H
#define ROOT 0
#include <vector>

typedef struct box {
    float3 center;
    float half_extent;
} box_t;

typedef struct node {
    int children;
    box_t box;
    float4 position;
    int next;
} node_t;

typedef struct octree {
    std::vector<node_t> nodes;
} octree_t;

void octree_init(octree_t *octree, float3 center, float half_extent) {
    octree->nodes = std::vector<node_t>();
    octree->nodes.push_back({ROOT, {center, half_extent}, {0, 0, 0, 0}, 0});
}

void octree_split(octree_t *octree, int node) {
    node_t parent = octree->nodes[node];

    float half = parent.box.half_extent;

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
        center.x += half/2 * (i & 1 ? 1 : -1);
        center.y += half/2 * (i & 2 ? 1 : -1);
        center.z += half/2 * (i & 4 ? 1 : -1);

        octree->nodes.push_back({ROOT,
                                 {center, half/2},
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
    for (int i = 0; i < 8; i++) {
        if (octree->nodes[octree->nodes[node].children + i].position.w == 0) {
            continue;
        }
        position.x += octree->nodes[octree->nodes[node].children + i].position.x * octree->nodes[octree->nodes[node].children + i].position.w;
        position.y += octree->nodes[octree->nodes[node].children + i].position.y * octree->nodes[octree->nodes[node].children + i].position.w;
        position.z += octree->nodes[octree->nodes[node].children + i].position.z * octree->nodes[octree->nodes[node].children + i].position.w;
        position.w += octree->nodes[octree->nodes[node].children + i].position.w;
    }

    if (position.w == 0) {
        return;
    }

    position.x /= position.w;
    position.y /= position.w;
    position.z /= position.w;
    octree->nodes[node].position = position;
}

void octree_insert(octree_t *octree, float4 position) {
    int current = ROOT;
    float half = octree->nodes[current].box.half_extent;
    float3 center = octree->nodes[current].box.center;
    // check if within bounds
    if (position.x < center.x - half ||
        position.x > center.x + half ||
        position.y < center.y - half ||
        position.y > center.y + half ||
        position.z < center.z - half ||
        position.z > center.z + half) {
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
    float theta_sq = theta * theta;
    int node = ROOT;
    float3 acceleration = {0, 0, 0};
    while (true) {
        node_t n = octree->nodes[node];
        float3 pos_to_pos2 = {n.position.x - position.x,
                              n.position.y - position.y,
                              n.position.z - position.z};
        float d_sq = pos_to_pos2.x * pos_to_pos2.x +
                     pos_to_pos2.y * pos_to_pos2.y +
                     pos_to_pos2.z * pos_to_pos2.z;
        // if is leaf or approximation criterion is true, calculate acceleration
        if (n.children == ROOT ||
            n.box.half_extent*n.box.half_extent < theta_sq*d_sq) {
            if (d_sq < 1e-4f) d_sq = 1e-4f;

            float inv_dist = 1.0f / sqrtf(d_sq);
            float inv_dist_cubed = inv_dist * inv_dist * inv_dist;
            float acc = G * n.position.w * position.w * inv_dist_cubed;

            acceleration.x += pos_to_pos2.x * acc / position.w;
            acceleration.y += pos_to_pos2.y * acc / position.w;
            acceleration.z += pos_to_pos2.z * acc / position.w;

            if (n.next == ROOT) {
                break;
            }
            node = n.next;
        } else {
            node = n.children;
        }
    }
    return acceleration;
}
#endif //NBODY_OCTREE_H
