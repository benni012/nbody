//
// Created by Ben Chwalek on 11.12.24.
//

#ifndef NBODY_OCTREE_H
#define NBODY_OCTREE_H
#define ROOT 0
#define LEAF_CAPACITY 16
#include <vector>
#include <functional>

typedef struct box {
    float3 center;
    float half_extent;
} box_t;

typedef struct node {
    int children;
    box_t box;
    float4 center_of_mass;
    int pos_idx, count;
    int next;
} node_t;

typedef struct octree {
    std::vector<node_t> nodes;
} octree_t;

void octree_init(octree_t *octree, float3 center, float half_extent) {
    octree->nodes = std::vector<node_t>();
    octree->nodes.push_back({ROOT, {center, half_extent}, {0, 0, 0, 0}, 0, 0, 0});
}

int find_split(body_t *bodies, int start, int end, std::function<bool(body_t)> func) {
    for (int i = start; i < end; i++) {
        if (func(bodies[i])) {
            return i;
        }
    }
}

void octree_split(octree_t *octree, int node, body_t *bodies) {
    node_t parent = octree->nodes[node];
//    fprintf(stderr, "%d\n", parent.count);
    float3 center = parent.box.center;
    int split[] = {parent.pos_idx, 0, 0, 0, 0, 0, 0, 0, parent.pos_idx+parent.count};

    split[4] = std::partition(bodies + split[0], bodies + split[8], [&center](body_t a) -> bool {
        return a.position.z < center.z;
    }) - bodies;

//    // sort bodies in-place by y and z < center.z
//    std::sort(bodies + split[0], bodies + split[4], [](body_t a, body_t b) {
//        return a.position.y < b.position.y;
//    });
//    // find where to split (where y > center.y) and z < center.z
//    split[2] = find_split(bodies, split[0], split[4], [&center](body_t a) -> bool {
//        return a.position.y > center.y;
//    });
//    // sort bodies in-place by y < center.y and z > center.z
//    std::sort(bodies + split[4], bodies + split[8], [](body_t a, body_t b) {
//        return a.position.y < b.position.y;
//    });
//    // find where to split (where y > center.y) and z > center.z
//    split[6] = find_split(bodies, split[4], split[8], [&center](body_t a) -> bool {
//        return a.position.y > center.y;
//    });
//
//
//    // sort bodies in-place by x and z < center.z
//    std::sort(bodies + split[0], bodies + split[2], [](body_t a, body_t b) {
//        return a.position.x < b.position.x;
//    });
//    // find where to split (where x > center.x) and z < center.z
//    split[1] = find_split(bodies, split[0], split[2], [&center](body_t a) -> bool {
//        return a.position.x > center.x;
//    });
//    // sort bodies in-place by x < center.x and z > center.z
//    std::sort(bodies + split[2], bodies + split[4], [](body_t a, body_t b) {
//        return a.position.x < b.position.x;
//    });
//    // find where to split (where x > center.x) and z < center.z
//    split[3] = find_split(bodies, split[2], split[4], [&center](body_t a) -> bool {
//        return a.position.x > center.x;
//    });
//    // sort bodies in-place by x and z > center.z
//    std::sort(bodies + split[4], bodies + split[6], [](body_t a, body_t b) {
//        return a.position.x < b.position.x;
//    });
//    // find where to split (where x > center.x) and z > center.z
//    split[5] = find_split(bodies, split[4], split[6], [&center](body_t a) -> bool {
//        return a.position.x > center.x;
//    });
//    // sort bodies in-place by x < center.x and z > center.z
//    std::sort(bodies + split[6], bodies + split[8], [](body_t a, body_t b) {
//        return a.position.x < b.position.x;
//    });
//    // find where to split (where x > center.x) and z > center.z
//    split[7] = find_split(bodies, split[6], split[8], [&center](body_t a) -> bool {
//        return a.position.x > center.x;
//    });

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
                                    split[i], split[i+1] - split[i],
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
    while (node < octree->nodes.size()) {
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
    octree_calculate_proxies(octree, ROOT);
}

float3 octree_calculate_acceleration(octree_t *octree, float4 position, body_t *bodies, float theta) {
    float theta_sq = theta * theta;
    int node = ROOT;
    float3 acceleration = {0, 0, 0};
    while (true) {
        node_t n = octree->nodes[node];
        if (n.children == ROOT) { // leaf
            for (int i = n.pos_idx; i < n.pos_idx+n.count; i++) {
                float4 other = bodies[i].position;
                float dx = other.x - position.x;
                float dy = other.y - position.y;
                float dz = other.z - position.z;
                float d_sq = dx * dx + dy * dy + dz * dz;
                if (d_sq < 1e-4f) d_sq = 1e-4f;

                float inv_dist = 1.0f / sqrtf(d_sq);
                float inv_dist_cubed = inv_dist * inv_dist * inv_dist;
                float acc = G * other.w * inv_dist_cubed;

                acceleration.x += dx * acc;
                acceleration.y += dy * acc;
                acceleration.z += dz * acc;
            }
            if (n.next == ROOT) {
                break;
            }
            node = n.next;
        } else {
            float dx = n.center_of_mass.x - position.x;
            float dy = n.center_of_mass.y - position.y;
            float dz = n.center_of_mass.z - position.z;
            float d_sq = dx * dx + dy * dy + dz * dz;
            if (n.box.half_extent * n.box.half_extent < theta_sq * d_sq) { // approximation criterion
                if (d_sq < 1e-4f) d_sq = 1e-4f;

                float inv_dist = 1.0f / sqrtf(d_sq);
                float inv_dist_cubed = inv_dist * inv_dist * inv_dist;
                float acc = G * n.center_of_mass.w * inv_dist_cubed;

                acceleration.x += dx * acc;
                acceleration.y += dy * acc;
                acceleration.z += dz * acc;

                if (n.next == ROOT) {
                    break;
                }
                node = n.next;
            } else {
                node = n.children;
            }
        }
    }
    return acceleration;
}
#endif //NBODY_OCTREE_H
//void octree_insert(octree_t *octree, float4 position) {
//    int current = ROOT;
//    float half = octree->nodes[current].box.half_extent;
//    float3 center = octree->nodes[current].box.center;
//    // check if within bounds
//    if (position.x < center.x - half ||
//        position.x > center.x + half ||
//        position.y < center.y - half ||
//        position.y > center.y + half ||
//        position.z < center.z - half ||
//        position.z > center.z + half) {
//        return;
//    }
//    while (octree->nodes[current].children != ROOT) {
//        int child = 0;
//        if (position.x > octree->nodes[current].box.center.x) child |= 1;
//        if (position.y > octree->nodes[current].box.center.y) child |= 2;
//        if (position.z > octree->nodes[current].box.center.z) child |= 4;
//        current = octree->nodes[current].children + child;
//    }
//    int leaf = current;
//    if (octree->nodes[leaf].position.w == 0) {
//        octree->nodes[leaf].position = position;
//    } else if (position.x == octree->nodes[leaf].position.x &&
//               position.y == octree->nodes[leaf].position.y &&
//               position.z == octree->nodes[leaf].position.z) {
//        octree->nodes[leaf].position.w += position.w;
//    } else {
//        octree_split(octree, leaf);
//        float4 old_position = octree->nodes[leaf].position;
//        octree_insert(octree, position);
//        octree_insert(octree, old_position);
//        octree->nodes[leaf].position = {0, 0, 0, 0};
//    }
//}
