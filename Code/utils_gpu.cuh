/**
 * @file utils_gpu.cuh
 * @author Sam McKeown (mckeowsa@tcd.ie)
 * @brief A header containg some macros and inline functions used in the GPU
 * code.
 * @version 1.0
 * @date 2022-04-03
 * 
 * 
 * 
 */

#ifndef UTILS_GPU_H__
#define UTILS_GPU_H__

#include "fp_type.h"

/**
 * @brief These timing macros can be used to measure times with CUDA events in a
 * (hopefully) clean way. They expect an array float timings[] to be declared
 * and have at least as many entries as times measured.
 * 
 * The functionality could be moved into function calls, but I found this way to
 * be less cluttered.
 */
#define TIME_INIT()          \
    cudaEvent_t start;       \
    cudaEvent_t end;         \
    cudaEventCreate(&start); \
    cudaEventCreate(&end);   \
    int timings_index = 0

#define TIME_START() \
    cudaEventRecord(start)

#define TIME_END()                                                 \
    cudaEventRecord(end);                                          \
    if (timings != NULL)                                           \
    {                                                              \
        cudaEventSynchronize(start);                               \
        cudaEventSynchronize(end);                                 \
        cudaEventElapsedTime(timings + timings_index, start, end); \
        timings[timings_index] /= 1000.0f;                         \
    }                                                              \
    timings_index++

#define TIME_FINISH()        \
    cudaEventDestroy(start); \
    cudaEventDestroy(end)

// inline void set_elapsed_time(cudaEvent_t start, cudaEvent_t end, float *const timings, int* index)
// {
//     if (timings != NULL)
//     {
//         cudaEventSynchronize(start);
//         cudaEventSynchronize(end);
//         cudaEventElapsedTime(timings + *index, start, end);
//         timings[*index] /= 1000.0f;
//     }
//     (*index)++;
//
//     return;
// }

/**
 * @brief Basic max and min macros.
 * 
 */
#define MAX(a, b) ((a) > (b)) ? (a) : (b)
#define MIN(a, b) ((a) < (b)) ? (a) : (b)

/**
 * @brief A function for dividing and rounding up. If a is negative we return 0.
 * This is useful when we have an expression like div_up(size - threadIdx.x,
 * stride), with threadIdx.x >= size.
 */
inline __host__ __device__ int div_up(int a, int b)
{
    if (a < 0)
    {
        return 0;
    }

    return (a / b) + ((a % b == 0) ? 0 : 1);
}

/**
 * @brief A function for determining how many entries should be allocated to a
 * given bin.
 * 
 * @param size The number of entries.
 * @param bins The number of bins (thread blocks in this case).
 * @param index The bin index.
 * @return int The allocated number of entries.
 */
inline __host__ __device__ int allocation(const int size, const int bins, const int index)
{
    const int quot = size / bins;
    const int rem = size % bins;

    if (index < rem)
    {
        return quot + 1;
    }
    else
    {
        return quot;
    }
}

/**
 * @brief A function for determining the offset into a vector at which an
 * allocation should start.
 * 
 * @param size The number of entries.
 * @param bins The number of bins (thread blocks in this case).
 * @param index The bin index.
 * @return int The offset at which the allocation starts.
 */
inline __host__ __device__ int offset(const int size, const int bins, const int index)
{
    const int quot = size / bins;
    const int rem = size % bins;

    if (index < rem)
    {
        return (quot + 1) * index;
    }
    else
    {
        return quot * index + rem;
    }
}

#endif // UTILS_GPU_H__