#ifndef AVERAGE_GPU_H__
#define AVERAGE_GPU_H__

#include <stdio.h>

#include "fp_type.h"

#include "device_props.h"

// This way CUDA code doesn't get included when compiling with gcc
#ifdef __CUDACC__
#include "utils_gpu.cuh"
#endif

// This can be used for tuning. 128 seems about ideal for the Turing.
#define MAX_ENTRIES_PER_THREAD                                                 \
  128 /* The maximum entries a thread will average */

#ifdef __cplusplus
extern "C" {
#endif
int average_rows_gpu(const int height, const int width, const int stride,
                     const FP_TYPE* const matrix, FP_TYPE* const averages,
                     const int block_size, float timings[5]);
#ifdef __cplusplus
}
#endif

#endif // AVERAGE_GPU_H__
