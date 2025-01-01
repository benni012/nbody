#ifndef NBODY_BODY_H
#define NBODY_BODY_H

#ifdef CUDA_FOUND
#include <cuda.h>
#include <cuda_runtime.h>
#endif 

typedef struct body {
    float4 position;
    float3 velocity;
} body_t;
#endif // NBODY_BODY_H
