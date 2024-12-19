//
// Created by Ben Chwalek on 10.12.24.
//

#ifndef NBODY_NBODY_CPU_H
#define NBODY_NBODY_CPU_H
#include "graphics.h"
void cpu_update_naive(int N, body_t *bodies) {
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            float dx = bodies[j].position.x - bodies[i].position.x;
            float dy = bodies[j].position.y - bodies[i].position.y;
            float dz = bodies[j].position.z - bodies[i].position.z;

            float distSq = dx * dx + dy * dy + dz * dz;
            if (distSq < 1e-4f) distSq = 1e-4f;

            float invDist = 1.0f / sqrtf(distSq);
            float invDist3 = invDist * invDist * invDist;

            float force = G * bodies[i].position.w * bodies[j].position.w * invDist3;

            bodies[i].velocity.x += dx * force / bodies[i].position.w;
            bodies[i].velocity.y += dy * force / bodies[i].position.w;
            bodies[i].velocity.z += dz * force / bodies[i].position.w;

            bodies[j].velocity.x -= dx * force / bodies[j].position.w;
            bodies[j].velocity.y -= dy * force / bodies[j].position.w;
            bodies[j].velocity.z -= dz * force / bodies[j].position.w;
        }
    }

    for (int i = 0; i < N; i++) {
        bodies[i].position.x += bodies[i].velocity.x;
        bodies[i].position.y += bodies[i].velocity.y;
        bodies[i].position.z += bodies[i].velocity.z;
    }
}
void cpu_update_bh(int N, body_t *bodies, octree_t *octree) {
    // Calculate accelerations
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        float3 acceleration = octree_calculate_acceleration(octree, bodies[i].position, bodies, 0.6);
        bodies[i].velocity.x += acceleration.x;
        bodies[i].velocity.y += acceleration.y;
        bodies[i].velocity.z += acceleration.z;
    }

    // Update positions
    for (int i = 0; i < N; i++) {
        bodies[i].position.x += bodies[i].velocity.x;
        bodies[i].position.y += bodies[i].velocity.y;
        bodies[i].position.z += bodies[i].velocity.z;
    }
}
#endif //NBODY_NBODY_CPU_H
