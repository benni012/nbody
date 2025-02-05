#ifndef NBODY_NBODY_CPU_H
#define NBODY_NBODY_CPU_H
#include "structures.h"

/**
 * @brief Update the bodies using the naive O(N^2) algorithm
 *
 * @param N Number of bodies
 * @param bodies Array of bodies
 */
void cpu_update_naive(int N, body_t *bodies) {
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            float dx = bodies[j].position.x - bodies[i].position.x;
            float dy = bodies[j].position.y - bodies[i].position.y;
            float dz = bodies[j].position.z - bodies[i].position.z;

            float distSq = dx * dx + dy * dy + dz * dz + EPS_SQ;

            float invDist = 1.0f / sqrtf(distSq);
            float invDist3 = invDist * invDist * invDist;

            float force =
                G * bodies[i].position.w * bodies[j].position.w * invDist3;

            bodies[i].velocity.x += dx * force / bodies[i].position.w * DT;
            bodies[i].velocity.y += dy * force / bodies[i].position.w * DT;
            bodies[i].velocity.z += dz * force / bodies[i].position.w * DT;

            bodies[j].velocity.x -= dx * force / bodies[j].position.w * DT;
            bodies[j].velocity.y -= dy * force / bodies[j].position.w * DT;
            bodies[j].velocity.z -= dz * force / bodies[j].position.w * DT;
        }
    }

    for (int i = 0; i < N; i++) {
        bodies[i].position.x += bodies[i].velocity.x * DT;
        bodies[i].position.y += bodies[i].velocity.y * DT;
        bodies[i].position.z += bodies[i].velocity.z * DT;
    }
}

/**
 * @brief Calculate the acceleration of a body using the Barnes-Hut algorithm
 *
 * @param octree Pointer to the octree
 * @param position Position of the body
 * @param bodies Array of bodies
 * @param theta Barnes-Hut criterion threshold
 * @return Acceleration of the body
 */
float3 octree_calculate_acceleration(octree_t *octree, float4 position,
                                     body_t *bodies, float theta) {
    float theta_sq = theta * theta;
    int node = ROOT;
    float3 acceleration = {0, 0, 0};
    // we use the next pointer to fully traverse the tree, taking shortcuts when
    // possible (i.e. when the approximation criterion is satisfied)
    while (true) {
        node_t n = octree->nodes[node];
        float dx = n.center_of_mass.x - position.x;
        float dy = n.center_of_mass.y - position.y;
        float dz = n.center_of_mass.z - position.z;
        float d_sq = dx * dx + dy * dy + dz * dz + EPS_SQ; // damping factor

        if (4 * n.box.half_extent * n.box.half_extent <
            theta_sq * d_sq) { // approximation criterion
            // approximate and go to next sibling
            float dc = sqrtf(d_sq) * d_sq;
            float acc = G * n.center_of_mass.w / dc;

            acceleration.x += dx * acc;
            acceleration.y += dy * acc;
            acceleration.z += dz * acc;

            if (n.next == ROOT) {
                break;
            }
            node = n.next;
        } else if (n.children == ROOT) { // leaf
            // calculate acceleration using all bodies in the leaf
            for (int i = n.pos_idx; i < n.pos_idx + n.count; i++) {
                float4 other = bodies[i].position;
                float dx = other.x - position.x;
                float dy = other.y - position.y;
                float dz = other.z - position.z;
                float d_sq =
                    dx * dx + dy * dy + dz * dz + EPS_SQ; // damping factor

                float dc = sqrtf(d_sq) * d_sq;
                float acc = G * other.w / dc;

                acceleration.x += dx * acc;
                acceleration.y += dy * acc;
                acceleration.z += dz * acc;
            }
            if (n.next == ROOT) {
                break;
            }
            node = n.next;
        } else {
            // go to the first child
            node = n.children;
        }
    }
    return acceleration;
}

void cpu_update_bh(int N, body_t *bodies, octree_t *octree) {
    // Calculate accelerations
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        float3 acceleration = octree_calculate_acceleration(
            octree, bodies[i].position, bodies, 1.0);
        bodies[i].velocity.x += acceleration.x * DT;
        bodies[i].velocity.y += acceleration.y * DT;
        bodies[i].velocity.z += acceleration.z * DT;

        bodies[i].position.x += bodies[i].velocity.x * DT;
        bodies[i].position.y += bodies[i].velocity.y * DT;
        bodies[i].position.z += bodies[i].velocity.z * DT;
    }
}
#endif // NBODY_NBODY_CPU_H
