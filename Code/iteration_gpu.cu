/**
 * @file iteration_gpu.cu
 * @author Sam McKeown (mckeowsa@tcd.ie)
 * @brief Implements a GPU function to perform the given iteration scheme. All
 * computation is done in shared memory, with adjacent blocks communicating
 * when necessary.
 * @version 2.0
 * @date 2022-04-08
 *
 *
 *
 */

#include "iteration_gpu.h"

/**
 * @brief A struct to store data communicated between blocks.
 *
 */
typedef struct {
  FP_TYPE values[2]; /* The values of the row on one side of the boundary */
  bool    ready;     /* Whether the values are up to date */
} Boundary;

/**
 * @brief An enum for describing which side of the block's allocation a
 * boundary is on.
 *
 */
enum Side { LEFT_SIDE = 0, RIGHT_SIDE = 1 };

/**
 * @brief An enum for describing whether a boundary is on the edge of a row.
 *
 */
enum Edge { LEFT_EDGE = -1, NOT_EDGE = 0, RIGHT_EDGE = 1 };

/**
 * @brief Copies two vectors on the device in a parallel way.
 *
 * @param size The length of the vectors.
 * @param src The source vector.
 * @param dest The destination vector.
 */
__device__ void copy_vector(const int size,
                            const FP_TYPE* const __restrict__ src,
                            FP_TYPE* const __restrict__ dest) {
  // thread_steps will be 0 if threadIdx.x is out of bounds.
  const int thread_steps = div_up(size - threadIdx.x, blockDim.x);

  for (int i = 0; i < thread_steps; i++) {
    dest[i * blockDim.x + threadIdx.x] = src[i * blockDim.x + threadIdx.x];
  }
}

/**
 * @brief The result of applying the iteration with the given 5 values.
 *
 * @param old_m2 The old value at index j - 2.
 * @param old_m1 The old value at index j - 1.
 * @param old The old value at index j.
 * @param old_p1 The old value at index j + 1.
 * @param old_p2 The old value at index j + 2.
 * @return FP_TYPE The result of the iteration.
 */
inline __device__ FP_TYPE iterate_point(const FP_TYPE old_m2,
                                        const FP_TYPE old_m1, const FP_TYPE old,
                                        const FP_TYPE old_p1,
                                        const FP_TYPE old_p2) {
  // Using float constants in the single precision case massively improves
  // performance.
#ifdef USE_DOUBLE
  return 0.34 * old_m2 + 0.28 * old_m1 + 0.2 * old + 0.12 * old_p1 +
         0.06 * old_p2;
#else
  // return 0.34 * old_m2 + 0.28 * old_m1 + 0.2 * old + 0.12 * old_p1 + 0.06 *
  // old_p2;
  return 0.34f * old_m2 + 0.28f * old_m1 + 0.2f * old + 0.12f * old_p1 +
         0.06f * old_p2;
#endif
}

/**
 * @brief Applies the iteration once to the entries on the interior of the
 * block's allocation (that is, all but the first and last two).
 *
 * This is the function executed by all but two of the threads in the block.
 *
 * @param interior_row The vector into which the newly calculated interior
 * values should be stored.
 * @param interior_row_swap A vector containing the previous iteration's
 * interior values.
 * @param thread_steps The number of entries the calling thread should
 * calculate.
 * @param thread_stride The gap between subsequent entries calculated by the
 * thread.
 */
__device__ void
    iterate_interior(FP_TYPE* const __restrict__ interior_row,
                     const FP_TYPE* const __restrict__ interior_row_swap,
                     const int thread_steps, const int thread_stride) {
  // We assume thread_steps is appropriately
  // calculated and won't bring us out of bounds.
  for (int i = 0; i < thread_steps; i++) {
    const int index     = i * thread_stride + threadIdx.x;
    interior_row[index] = iterate_point(
        interior_row_swap[index - 2], interior_row_swap[index - 1],
        interior_row_swap[index], interior_row_swap[index + 1],
        interior_row_swap[index + 2]);
  }

  return;
}

/**
 * @brief Handles the special cases on the boundaries of the block's allocation
 * (that is, the first and last two entries). The communication with other
 * blocks is all done here.
 *
 * I've tried to follow Nvidia's recommendations for inter-block communication,
 * where we use volatile variables in global memory to pass data around and use
 * __threadfence() before setting any flags.
 *
 * Should be called by two threads, one for the left boundary and one for the
 * right.
 *
 * @param thread_values A pointer to the first of the two boundary values to
 * be calculated by this thread.
 * @param thread_values_swap A pointer to the first of the two old values.
 * @param out_bound Written to by this thread and read by a thread in the
 * neighbouring block.
 * @param in_bound Read by this thread and written to by a thread in the
 * neighbouring block.
 * @param edge Describes whether the thread's values are on an edge of the
 * matrix.
 * @param edge_values The values of the first to entries in the row. Can be
 * NULL if the thread's values aren't on an edge.
 */
__device__ void
    iterate_boundary(FP_TYPE* const __restrict__ thread_values,
                     const FP_TYPE* const __restrict__ thread_values_swap,
                     volatile Boundary* const __restrict__ out_bound,
                     volatile Boundary* const __restrict__ in_bound,
                     const Edge edge, const FP_TYPE edge_values[2]) {
  /* Exit if on left edge */

  // If we're on the left edge there's nothing to update or communicate.
  if (edge == LEFT_EDGE) {
    return;
  }

  /* Declare variables */

  const Side side =
      (threadIdx.x % 2 == 0)
          ? LEFT_SIDE
          : RIGHT_SIDE; /* The side of the block's allocation we're on */
  FP_TYPE
  in_values[2]; /* Holds the values from outside the block's allocation*/

  /* Communicate and read in in_values */

  if (edge == NOT_EDGE) /* If we're not on an edge we need to communicate with
                           another block */
  {
    /* Update out_bound */

    // We shouldn't touch out_bound until
    // the other block has read from it.
    while (out_bound->ready) {
    }

    // We update it with the old values from this thread.
    out_bound->values[0] = thread_values_swap[0];
    out_bound->values[1] = thread_values_swap[1];

    // A threadfence ensures the new values are seen before the flag.
    __threadfence();
    out_bound->ready = true;

    /* Read from in_bound */

    // We should wait until the values have been updated.
    while (!in_bound->ready) {
    }

    in_values[0] = in_bound->values[0];
    in_values[1] = in_bound->values[1];

    in_bound->ready = false;
  } else /* If we're on an edge we just use the edge_values */
  {
    in_values[0] = edge_values[0];
    in_values[1] = edge_values[1];
  }

  /* Iterate values and return */

  switch (side) {
    case LEFT_SIDE:
      thread_values[0] =
          iterate_point(in_values[0], in_values[1], thread_values_swap[0],
                        thread_values_swap[1], thread_values_swap[2]);

      thread_values[1] = iterate_point(
          in_values[1], thread_values_swap[0], thread_values_swap[1],
          thread_values_swap[2], thread_values_swap[3]);
      break;

    case RIGHT_SIDE:
      // Negative indicies are alright because thread_values_swap
      // is at the end of a larger array.
      thread_values[0] = iterate_point(
          thread_values_swap[-2], thread_values_swap[-1], thread_values_swap[0],
          thread_values_swap[1], in_values[0]);

      thread_values[1] =
          iterate_point(thread_values_swap[-1], thread_values_swap[0],
                        thread_values_swap[1], in_values[0], in_values[1]);
      break;
  }

  return;
}

/**
 * @brief When called by every thread in the block, has the effect of iterating
 * the values in the block's allocation. It calls one of the above iteration
 * functions after working out the appropriate arguments for the calling thread.
 *
 * Since the variables declared in this function don't change between calls, we
 * could save time by computing them once and passing them as arguments. It
 * seemed cleaner to do it this way, but might make a noticeable difference for
 * small grids iterated many times (though they'd converge quickly anyway).
 *
 * @param local_width The number of entries in the block's allocation.
 * @param block_row A pointer to the vector to store the results.
 * @param block_row_swap A pointer to the old values.
 * @param out_bounds The two Boundary structs used for outgoing communication.
 * Can be NULL if not needed by the calling thread.
 * @param in_bounds The two Boundary structs used for incoming communication.
 * Can be NULL if not needed by the calling thread.
 * @param edges Describes the two edges of the block's allocation.
 * Can be NULL if not needed by the calling thread.
 * @param edge_values The first two values in the row.
 * Can be NULL if not needed by the calling thread.
 */
__device__ void iterate_row_block(const int                local_width,
                                  FP_TYPE* const           block_row,
                                  FP_TYPE* const           block_row_swap,
                                  volatile Boundary* const out_bounds[2],
                                  volatile Boundary* const in_bounds[2],
                                  const Edge               edges[2],
                                  const FP_TYPE            edge_values[2]) {
  const int interior_threads =
      blockDim.x - 2; /* The number of threads doing the standard iteration */

  if (threadIdx.x < interior_threads) /* If we're in the interior we call
                                         iterate_interior() */
  {
    const int interior_points =
        local_width - 4; /* The number of non-boundary entries */

    FP_TYPE* const interior_row =
        block_row + 2; /* Pointer to the first interior entry in the output */
    FP_TYPE* const interior_row_swap =
        block_row_swap +
        2; /* Pointer to the first interior entry in the old values */

    const int thread_steps = div_up(
        interior_points - threadIdx.x,
        interior_threads); /* The number of entries for the thread to iterate */

    iterate_interior(interior_row, interior_row_swap, thread_steps,
                     interior_threads);
  }

  else /* If the thread is handling a boundary we call iterate_boundary() */
  {
    const Side side = (threadIdx.x % 2 == 0)
                          ? LEFT_SIDE
                          : RIGHT_SIDE; /* The side of the block's allocation
                                           the thread handles */
    const Edge edge =
        edges[side]; /* Describes whether the thread handles an edge */

    const int thread_offset =
        (side == LEFT_SIDE)
            ? 0
            : local_width - 2; /* Offset for the entries the thread handles */
    FP_TYPE* const thread_values =
        block_row +
        thread_offset; /* Pointer to the first entry for the thread's output */
    FP_TYPE* const thread_values_swap =
        block_row_swap + thread_offset; /* Pointer to the first entry for the
                                           thread's old values */

    iterate_boundary(thread_values, thread_values_swap, out_bounds[side],
                     in_bounds[side], edge, edge_values);
  }

  return;
}

/**
 * @brief Returns the initial value for a given entry.
 *
 * @param height The height of the matrix.
 * @param row The row on the entry.
 * @param col The column of the entry.
 * @return FP_TYPE The initial value for the entry.
 */
inline __device__ FP_TYPE initial_value(const int height, const int row,
                                        const int col) {
  switch (col) {
    // Using different literals is pedantic but seemed more consistent.
#ifdef USE_DOUBLE
    case 0: return (FP_TYPE) (row + 1) / (FP_TYPE) (height);
    case 1: return 0.8 * (FP_TYPE) (row + 1) / (FP_TYPE) (height);
    default: return 1.0 / 15360.0;
#else
    case 0: return (FP_TYPE) (row + 1) / (FP_TYPE) (height);
    case 1: return 0.8f * (FP_TYPE) (row + 1) / (FP_TYPE) (height);
    default: return 1.0f / 15360.0f;
#endif
  }
}

/**
 * @brief Sets entries to their initial values in parallel.
 *
 * @param height The height of the matrix.
 * @param row The row the block's allocation is on.
 * @param block_offset The offset into the row at which the block's allocation
 * starts.
 * @param block_width The width of the block's allocation.
 * @param block_row A pointer to the first entry to initialise.
 */
__device__ void set_initial_conditions_block(const int height, const int row,
                                             const int      block_offset,
                                             const int      block_width,
                                             FP_TYPE* const block_row) {
  // If threadIdx.x is out of bounds this will be 0.
  const int thread_steps = div_up(block_width - threadIdx.x, blockDim.x);

  for (int i = 0; i < thread_steps; i++) {
    const int index = i * blockDim.x + threadIdx.x;
    const int col   = block_offset + index;

    block_row[index] = initial_value(height, row, col);
  }
}

/**
 * @brief A kernel to apply the given number of iterations to the initial
 * values.
 *
 * @param iterations The number of iterations.
 * @param height The height of the matrix.
 * @param width The width of the matrix.
 * @param matrix_gpu The matrix to store the output. Nothing is read from it.
 * @param blocks_per_row The number of blocks a row is divided across.
 * @param bounds A pointer to the global memory used for inter-block
 * communication.
 */
__global__ void apply_iterations_kernel(const int iterations, const int height,
                                        const int         width,
                                        FP_TYPE* const    matrix_gpu,
                                        const int         blocks_per_row,
                                        volatile Boundary bounds[]) {
  /* Grid-level constants */

  const int entries_per_block = div_up(
      width, blocks_per_row); /* The most entries allocated to a given block */
  const int rows_per_grid =
      gridDim.x /
      blocks_per_row; /* The number of rows being worked on at once */

  /* Block-level constants */

  // These would be the x, y coordinates if we used a 2D grid.
  const int start_row =
      blockIdx.x / blocks_per_row; /* The row a block starts on */
  const int block_row_index =
      blockIdx.x % blocks_per_row; /* The index among blocks on the same row */

  // block_width is entries_per_block or 1 less.
  const int block_width = allocation(
      width, blocks_per_row,
      block_row_index); /* The number of entries allocated to the block */
  const int block_offset = offset(
      width, blocks_per_row, block_row_index); /* The column of the first entry
                                                  allocated to the block */

  const int block_steps = div_up(
      height - start_row, rows_per_grid); /* The number of rows to iterate */

  const Edge edges[2] = {(block_row_index == 0) ? LEFT_EDGE : NOT_EDGE,
                         (block_row_index == blocks_per_row - 1)
                             ? RIGHT_EDGE
                             : NOT_EDGE}; /* Describes whether the block's
                                             allocation lies on an edge */

  /* Set up pointers to shared memory */

  extern __shared__ FP_TYPE
      block_shared[]; /* A pointer to the start of the dynamically allocated
                         shared memory */

  FP_TYPE* const block_row =
      block_shared + 0; /* The vector to store the finished values */
  FP_TYPE* const block_row_swap =
      block_row + entries_per_block; /* A vector to store itermediary values */

  /* Set up pointers to global memory */

  volatile Boundary* in_bounds[2] = {
      (edges[LEFT_SIDE] == LEFT_EDGE) ? NULL : bounds + 2 * blockIdx.x - 1,
      (edges[RIGHT_SIDE] == RIGHT_EDGE)
          ? NULL
          : bounds + 2 * blockIdx.x +
                2}; /* Used for incoming communication on the bounds */

  volatile Boundary* out_bounds[2] = {
      (edges[LEFT_SIDE] == LEFT_EDGE) ? NULL : bounds + 2 * blockIdx.x,
      (edges[RIGHT_SIDE] == RIGHT_EDGE)
          ? NULL
          : bounds + 2 * blockIdx.x +
                1}; /* Used for outgoing communication on the bounds */

  /* Loop over assigned rows and iterate them */

  for (int i = 0; i < block_steps; i++) {
    /* Set up row-specific values */

    const int row = i * rows_per_grid + start_row; /* The current row */

    const FP_TYPE edge_values[2] = {
        initial_value(height, row, 0),
        initial_value(height, row,
                      1)}; /* The first values on the current row */

    /* Set initial conditions */

    // We should sync here because we might still be copying from a previous
    // loop. Strictly, the entries are partitioned between threads in the
    // same way between this function and the copying function, so it
    // shouldn't lead to race conditions, but that could easily change if we
    // modify the functions.
    __syncthreads();
    set_initial_conditions_block(height, row, block_offset, block_width,
                                 block_row);
    set_initial_conditions_block(height, row, block_offset, block_width,
                                 block_row_swap);

    /* Perform iterations */

    if (iterations % 2 != 0) {
      __syncthreads();
      iterate_row_block(block_width, block_row, block_row_swap, out_bounds,
                        in_bounds, edges, edge_values);
    }
    for (int iter = 0; iter < (iterations / 2); iter++) {
      __syncthreads();
      iterate_row_block(block_width, block_row_swap, block_row, out_bounds,
                        in_bounds, edges, edge_values);

      __syncthreads();
      iterate_row_block(block_width, block_row, block_row_swap, out_bounds,
                        in_bounds, edges, edge_values);
    }

    /* Copy the result into the output matrix */

    FP_TYPE* const offset_matrix =
        matrix_gpu + row * width +
        block_offset; /* A pointer to the block's share of the row in global
                         memory */

    __syncthreads();
    copy_vector(block_width, block_row, offset_matrix);
  }

  return;
}

/**
 * @brief Uses the GPU to apply the given number of iterations to under the
 * scheme in the assignment.
 *
 * For small widths, each block will store a row of the matrix in shared memory
 * and iterate it. For larger matrices, rows will be split between blocks and
 * the necessary updates communicated through global memory.
 *
 * The block size should be 2 mod 32, so that the communication is done in a
 * seperate warp and the latency hidden.
 *
 * @param iterations The number of iterations.
 * @param height The height of the matrix.
 * @param width The width of the matrix.
 * @param matrix_gpu The matrix to store the output. Nothing is read from it.
 * @param block_size The block size to use. Should be 2 mod 32. 258 seems ideal.
 * @param timings An array into which to store the timings. Will be ignored if
 * set to NULL.
 * @return int Returns 0. Exits on failure.
 */
int apply_iterations_gpu(const int iterations, const int height,
                         const int width, FP_TYPE* const matrix,
                         const int block_size, float timings[5]) {
  TIME_INIT();

  /* Set up grid dimensions */

  TIME_START();
  const int max_shared_size =
      current_max_shared_per_block(); /* The most shared memory per block
                                         supported by the card */
  const int max_entries_per_block =
      max_shared_size / (2 * sizeof(FP_TYPE)); /* The largest row that can fit
                                                  in a block's shared memory */
  // const int max_entries_per_block = 5; /* Useful for debugging */

  // This would be the x dimension if we used a 2D grid.
  const int blocks_per_row =
      div_up(width, max_entries_per_block); /* The number of blocks that will
                                               work on a given row */
  const int entries_per_block = div_up(
      width, blocks_per_row); /* The most entries allocated to a given block */
  const int block_allocation_size =
      entries_per_block * 2 *
      sizeof(FP_TYPE); /* The size of each block's dynamically allocated shared
                          memory */

  // For best performance we can give the GPU as many blocks to work with as
  // possible.
  const int rows_per_grid = height; /* An upper bound on the number of rows for
                                       the grid to work on at once */
  // const int rows_per_grid = 2; /* Useful for debugging */

  // The following commented lines can be uncommented to verify assumptions
  // that this implementation makes. The card doesn't seem to have problems
  // even when they don't hold, and in particular limiting the number of
  // blocks decreases performance substantially.

  // const int sm_count = current_multiprocessor_count();               /* The
  // number of SMs on the selected card */ const int max_entries_per_grid =
  // max_entries_per_block * sm_count; /* The number of entries that can fit in
  // the card's shared memory */

  // This implementation assumes that at least one row can fit split across
  // the shared memory of the blocks. However the card seems to manage larger
  // widths, presumably by swapping out to main memory.
  //  if(width > max_entries_per_grid)
  //  {
  //      fprintf(stderr, "Too many columns to fit in shared memory.
  //      Exiting.\n"); exit(EXIT_FAILURE);
  //  }

  // It might be better to choose rows_per_grid such that all of the blocks
  // can be running simultaneously. Otherwise it's conceivable that we'll have
  // a block waiting on another one that isn't running.
  // In practice this doesn't seem to happen.
  //  const int entries_per_row = entries_per_block * blocks_per_row; /* The
  //  number of entries stored in shared memory for a row (might be more than
  //  width) */ const int rows_per_grid = MIN(height, max_entries_per_grid /
  //  entries_per_row); /* An upper bound on the number of rows for the grid to
  //  work on at once */

  // We choose the grid size to be a multiple of blocks_per_row
  const int max_grid_size = (current_max_grid_dim_x() / blocks_per_row) *
                            blocks_per_row; /* The maximum usable grid size */
  const int grid_size = MIN(
      max_grid_size, blocks_per_row * rows_per_grid); /* The grid size to use */

  if (grid_size <= 0) {
    fprintf(stderr, "Required grid too large. Exiting.\n");
    exit(EXIT_FAILURE);
  }

  // Useful for debugging
  //  printf("Row decomp:\t%d = %d x %d\n", width, blocks_per_row,
  //  entries_per_block); printf("Grid dims:\t%d x %d\n", grid_size,
  //  block_size);

  /* If supported, use larger shared memory */

  // I don't think this check can be done at compile time in host code.
  if (current_compute_capability() >= 700) {
    cudaFuncSetAttribute(apply_iterations_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_size);
  }
  TIME_END();

  /* Allocate memory on GPU */

  TIME_START();
  FP_TYPE* matrix_gpu = NULL; /* Will store the output matrix */
  if (cudaMalloc((void**) (&matrix_gpu), height * width * sizeof(FP_TYPE)) !=
      cudaSuccess) {
    fprintf(stderr, "Failed to allocate memory on device. Exiting.\n");
    exit(EXIT_FAILURE);
  }

  Boundary* bounds_gpu = NULL; /* Will be used to pass data across blocks */
  if (cudaMalloc((void**) (&bounds_gpu), 2 * grid_size * sizeof(Boundary)) !=
      cudaSuccess) {
    fprintf(stderr, "Failed to allocate memory on device. Exiting.\n");
    exit(EXIT_FAILURE);
  }
  TIME_END();

  /* Zero out inter-block memory */

  // This memory stores the flags used in communication. We need to set them
  // to false before we run any device code that depends on them.
  // This is the only operation we do from the host. Everything else is done
  // by the device.
  TIME_START();
  cudaMemset(bounds_gpu, 0, 2 * grid_size * sizeof(Boundary));
  TIME_END();

  /* Call kernel */

  TIME_START();
  apply_iterations_kernel<<<grid_size, block_size, block_allocation_size>>>(
      iterations, height, width, matrix_gpu, blocks_per_row, bounds_gpu);
  TIME_END();

  /* Copy results to host */

  TIME_START();
  cudaMemcpy(matrix, matrix_gpu, height * width * sizeof(FP_TYPE),
             cudaMemcpyDeviceToHost);
  TIME_END();

  /* Tidy up and return */

  cudaFree(matrix_gpu);
  cudaFree(bounds_gpu);

  TIME_FINISH();

  return 0;
}
