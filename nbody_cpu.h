//
// Created by Ben Chwalek on 10.12.24.
//

#ifndef NBODY_NBODY_CPU_H
#define NBODY_NBODY_CPU_H
#include "graphics.h"
void cpu_update_naive(int N, float4 *positions, float3 *velocities) {
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            float dx = positions[j].x - positions[i].x;
            float dy = positions[j].y - positions[i].y;
            float dz = positions[j].z - positions[i].z;

            float distSq = dx * dx + dy * dy + dz * dz;
            if (distSq < 1e-4f) distSq = 1e-4f;

            float invDist = 1.0f / sqrtf(distSq);
            float invDist3 = invDist * invDist * invDist;

            float force = G * positions[i].w * positions[j].w * invDist3;

            velocities[i].x += dx * force / positions[i].w;
            velocities[i].y += dy * force / positions[i].w;
            velocities[i].z += dz * force / positions[i].w;

            velocities[j].x -= dx * force / positions[j].w;
            velocities[j].y -= dy * force / positions[j].w;
            velocities[j].z -= dz * force / positions[j].w;
        }
    }

    for (int i = 0; i < N; i++) {
        positions[i].x += velocities[i].x;
        positions[i].y += velocities[i].y;
        positions[i].z += velocities[i].z;
    }
}
void cpu_update_bh(int N, float4 *positions, float3 *velocities, octree_t *octree) {
    // Calculate accelerations
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        float3 acceleration = octree_calculate_acceleration(octree, positions[i], .9);
        velocities[i].x += acceleration.x;
        velocities[i].y += acceleration.y;
        velocities[i].z += acceleration.z;
    }

    // Update positions
    for (int i = 0; i < N; i++) {
        positions[i].x += velocities[i].x;
        positions[i].y += velocities[i].y;
        positions[i].z += velocities[i].z;
    }
}
#endif //NBODY_NBODY_CPU_H
