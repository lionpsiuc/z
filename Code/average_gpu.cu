/**
 * @file average_gpu.cu
 * @author Sam McKeown (mckeowsa@tcd.ie)
 * @brief Implements a GPU function for averaging the rows of a matrix.
 * @version 1.0
 * @date 2022-04-03
 * 
 * 
 * 
 */

#include "average_gpu.h"

//This is the definition given in the Nvidia documentation, copy and pasted. It
//was discussed in lecture, so I thought it'd be alright to use it as-is.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(double *address, double val)
{
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                                             __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

/**
 * @brief A kernel to average the rows of a matrix. Each block will average a
 * portion of the entries in a given row, as determined by the width of the 
 * matrix and the value of MAX_ENTRIES_PER_THREAD. If there are more rows than
 * can be covered by the number of blocks, some blocks will work on multiple
 * rows.
 * 
 * @param height The height of the matrix.
 * @param width The width of the matrix.
 * @param stride The gap between the start of adjacent rows. Should not be less
 * than width.
 * @param matrix_gpu The matrix to work on (stored in the GPU's memory).
 * @param averages_gpu The vector (on the GPU) into to which the results are 
 * written.
 */
static __global__ void average_rows_kernel(const int height, const int width, const int stride,
                                           const FP_TYPE *const matrix_gpu, FP_TYPE *const averages_gpu)
{
    /* Declare shared memory */

    extern __shared__ FP_TYPE thread_averages[]; /* Averages computed by each thread */
    __shared__ FP_TYPE block_average;            /* The sum thread_averages across the block */

    /* Grid-level constants */

    //These would be the x, y dimensions if we used a 2D grid.
    const int blocks_per_row = div_up(width, MAX_ENTRIES_PER_THREAD * blockDim.x); /* The number of blocks working on a given row */
    const int rows_per_grid = gridDim.x / blocks_per_row;                          /* The number of rows the grid can work on at once */

    //Precomputing the inverse doesn't seem to give any improvement.
    // const FP_TYPE width_inv = (FP_TYPE)1 / width; /* The inverse of width */

    /* Block-level constants */

    //These would be the x, y coordinates if we used a 2D grid.
    const int start_row = blockIdx.x / blocks_per_row;                 /* The row a block starts on */
    const int block_row_index = blockIdx.x % blocks_per_row;           /* The index among blocks on the same row */
    const int block_steps = div_up(height - start_row, rows_per_grid); /* The number of rows to average */

    /* Thread-level constants */

    const int thread_row_id = block_row_index * blockDim.x + threadIdx.x;  /* The index among threads on the same row */
    const int thread_stride = blocks_per_row * blockDim.x;                 /* The gap between subsequent entries to average */
    const int thread_steps = div_up(width - thread_row_id, thread_stride); /* The number of entries to average per row */

    /* Loop over assigned rows and average them */

    //If start_row is out of bounds, block_steps will be 0 and nothing is done
    for (int i = 0; i < block_steps; i++)
    {
        /* Initialise block_average */

        if (threadIdx.x == 0)
        {
            block_average = 0.0f;
        }

        /* Compute offsets for row */

        const int row = i * rows_per_grid + start_row; /* The current row */

        const int block_offset = stride * row + blockDim.x * block_row_index; /* The first element for thread 0 to work on */
        const int thread_offset = block_offset + threadIdx.x;                 /* The first element for the thread to work on */

        /* Compute averages for each thread's elements */

        //If thread_offset is out of bounds, thread_steps will be 0
        thread_averages[threadIdx.x] = 0.0f;
        for (int i = 0; i < thread_steps; i++)
        {
            thread_averages[threadIdx.x] += matrix_gpu[i * thread_stride + thread_offset];
        }
        thread_averages[threadIdx.x] /= width;
        // thread_averages[threadIdx.x] *= width_inv;

        /* Reduce thread_averages into block_average */

        //We need to sync so that block_average will be initialised.
        __syncthreads();

        //The Turing supports block-level atomics, so those are used if possible.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
        atomicAdd(&block_average, thread_averages[threadIdx.x]);
#else
        atomicAdd_block(&block_average, thread_averages[threadIdx.x]);
#endif

        /* Reduce block_average into global row average */

        //We sync to make sure the block-level reduction is finished.
        __syncthreads();
        if (threadIdx.x == 0)
        {
            atomicAdd(averages_gpu + row, block_average);
        }
    }

    return;
}
/**
 * @brief A function to average the rows of a matrix on the GPU. Each block
 * will average a portion of the entries in a given row, as determined by
 * the width of the  matrix and the value of MAX_ENTRIES_PER_THREAD. If there 
 * are more rows than can be covered by the number of blocks, some blocks will
 * work on multiple rows.
 * 
 * @param height The height of the matrix.
 * @param width The width of the matrix.
 * @param stride The gap between the start of adjacent rows. Should not be less
 * than width.
 * @param matrix The matrix whose rows we average.
 * @param averages The vector into to which the results are written.
 * @param block_size The block size to use in the kernel. 32 seems ideal.
 * @param timings An array into which to store the timings. Will be ignored if
 * set to NULL.
 * @return int Returns 0. Exits on failure.
 */
int average_rows_gpu(const int height, const int width, const int stride,
                     const FP_TYPE *const matrix, FP_TYPE *const averages,
                     const int block_size, float timings[5])
{
    TIME_INIT();

    TIME_START();
    if (stride < width)
    {
        fprintf(stderr, "Bad call to average_rows_gpu() (width = %d, stride = %d). Exiting.\n", width, stride);
        exit(EXIT_FAILURE);
    }

    /* Set up grid dimensions */

    const int blocks_per_row = div_up(width, MAX_ENTRIES_PER_THREAD * block_size); /* The number of blocks that will work on a given row */
    const int block_allocation_size = block_size * sizeof(FP_TYPE);                /* The size of each block's dynamically allocated shared memory */

    //We choose this to be a multiple of blocks_per_row
    const int max_grid_size = (current_max_grid_dim_x() / blocks_per_row) * blocks_per_row; /* The maximum x dimension of a grid */
    const int grid_size = MIN(max_grid_size, blocks_per_row * height);                      /* The x dimension to use */

    // printf("Row decomp:\t%d = %d x %d\n", width, blocks_per_row, MAX_ENTRIES_PER_THREAD * block_size);
    // printf("Grid dims:\t%d x %d\n", grid_size, block_size);
    TIME_END();

    /* Allocate memory on the GPU */

    TIME_START();
    FP_TYPE *matrix_gpu = NULL; /* Will store a copy of matrix */
    if (cudaMalloc((void **)(&matrix_gpu), height * stride * sizeof(FP_TYPE)) != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate memory on device. Exiting.\n");
        exit(EXIT_FAILURE);
    }

    FP_TYPE *averages_gpu = NULL; /* Will store the output */
    if (cudaMalloc((void **)(&averages_gpu), height * sizeof(FP_TYPE)) != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate memory on device. Exiting.\n");
        exit(EXIT_FAILURE);
    }
    TIME_END();

    /* Copy matrix onto GPU */

    TIME_START();
    cudaMemcpy(matrix_gpu, matrix, height * stride * sizeof *matrix_gpu, cudaMemcpyHostToDevice);
    TIME_END();

    /* Launch kernel */

    TIME_START();
    average_rows_kernel<<<grid_size, block_size, block_allocation_size>>>(height, width, stride, matrix_gpu, averages_gpu);
    TIME_END();

    /* Copy the results back */

    TIME_START();
    cudaMemcpy(averages, averages_gpu, height * sizeof(FP_TYPE), cudaMemcpyDeviceToHost);
    TIME_END();

    /* Tidy up */

    cudaFree(matrix_gpu);
    cudaFree(averages_gpu);

    TIME_FINISH();

    return 0;
}
