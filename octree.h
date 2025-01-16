#ifndef NBODY_OCTREE_H
#define NBODY_OCTREE_H

#include <vector>
#include <functional>
#include <algorithm>
#include "structures.h"


void octree_init(octree_t *octree, float3 center, float half_extent, int max_nodes) {
    octree->nodes = new node_t[max_nodes];  // Allocate memory for the nodes
    octree->num_nodes = 1;
    octree->max_nodes = max_nodes;

    // Initialize the root node
    octree->nodes[ROOT] = {ROOT, {center, half_extent}, {0, 0, 0, 0}, 0, 0, 0};
}

void octree_split(octree_t *octree, int node, body_t *bodies) {
    node_t parent = octree->nodes[node];
    float3 center = parent.box.center;
    
    int split[] = {parent.pos_idx, 0, 0, 0, 0, 0, 0, 0, parent.pos_idx + parent.count};

    split[4] = std::partition(bodies + split[0], bodies + split[8], [&center](body_t a) -> bool {
        return a.position.z < center.z;
    }) - bodies;
    split[2] = std::partition(bodies + split[0], bodies + split[4], [&center](body_t a) -> bool {
        return a.position.y < center.y;
    }) - bodies;
    split[6] = std::partition(bodies + split[4], bodies + split[8], [&center](body_t a) -> bool {
        return a.position.y < center.y;
    }) - bodies;
    split[1] = std::partition(bodies + split[0], bodies + split[2], [&center](body_t a) -> bool {
        return a.position.x < center.x;
    }) - bodies;
    split[3] = std::partition(bodies + split[2], bodies + split[4], [&center](body_t a) -> bool {
        return a.position.x < center.x;
    }) - bodies;
    split[5] = std::partition(bodies + split[4], bodies + split[6], [&center](body_t a) -> bool {
        return a.position.x < center.x;
    }) - bodies;
    split[7] = std::partition(bodies + split[6], bodies + split[8], [&center](body_t a) -> bool {
        return a.position.x < center.x;
    }) - bodies;

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

void octree_calculate_proxies(octree_t *octree, int node) {
    if (octree->nodes[node].children == ROOT) {
        return;
    }
    for (int i = 0; i < 8; i++) {
        octree_calculate_proxies(octree, octree->nodes[node].children + i);
    }

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

    if (center_of_mass.w == 0) {
        return;
    }

    center_of_mass.x /= center_of_mass.w;
    center_of_mass.y /= center_of_mass.w;
    center_of_mass.z /= center_of_mass.w;
    octree->nodes[node].center_of_mass = center_of_mass;
}

void octree_build(octree_t *octree, body_t *bodies, int N) {
    octree->nodes[ROOT].pos_idx = 0;
    octree->nodes[ROOT].count = N;

    int node = 0;
    while (node < octree->num_nodes) {
        if (octree->nodes[node].count > LEAF_CAPACITY) {
            octree_split(octree, node, bodies);
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
    }
}

void octree_free(octree_t *octree) {
    if (octree->nodes != nullptr) {
        delete[] octree->nodes;  // Deallocate the memory for nodes
        octree->nodes = nullptr; // Set pointer to nullptr to avoid dangling references
    }
    octree->num_nodes = 0;        // Reset node count
    octree->max_nodes = 0;        // Reset max nodes
}

#endif //NBODY_OCTREE_H
