#ifndef ITERATION_GPU_H__
#define ITERATION_GPU_H__

#include <stdio.h>
#include <stdlib.h>

#include "fp_type.h"

#include "device_props.h"

// This way CUDA code doesn't get included when compiling with gcc
#ifdef __CUDACC__
#include "utils_gpu.cuh"
#endif

#ifdef __cplusplus
extern "C" {
#endif
int apply_iterations_gpu(const int iterations, const int height,
                         const int width, FP_TYPE* const matrix,
                         const int block_size, float timings[5]);
#ifdef __cplusplus
}
#endif

#endif // ITERATION_GPU_H__
