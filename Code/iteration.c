/**
 * @file iteration.c
 * @author Sam McKeown (mckeowsa@tcd.ie)
 * @brief Implements a CPU function for performing the given iteration scheme.
 * @version 1.0
 * @date 2022-04-04
 * 
 * 
 * 
 */

#include "iteration.h"

/**
 * @brief Sets a matrix to the initial conditions. Assumes there are extra
 * columns for facilitating the wrapping at edges.
 * 
 * @param height The height of the matrix.
 * @param width The width (excluding the extra colums) of the matrix.
 * @param matrix The matrix to be set. Should be have height X (width + 2)
 * entries allocated.
 */
static void set_initial_conditions(const int height, const int width,
                                   FP_TYPE *const matrix)
{
    //We assume there are two extra columns to simplify the wrapping around.
    const int stride = width + 2; /* The stride of matrix */

    /* Loop over rows and set initial values */

    for (int i = 0; i < height; i++)
    {
        /* Set edges */

        matrix[i * stride + 0] = (float)(i + 1) / (float)(height);
        matrix[i * stride + 1] = 0.8f * (float)(i + 1) / (float)(height);

        /* Set interior points */

        for (int j = 2; j < width; j++)
        {
            matrix[i * stride + j] = 1.0f / 15360.0f;
        }

        /* Set extra columns */

        matrix[i * stride + width + 0] = matrix[i * stride + 0];
        matrix[i * stride + width + 1] = matrix[i * stride + 1];
    }

    return;
}

/**
 * @brief Loops through the entries of a row and applies the iteration step.
 * 
 * @param width The width of the row.
 * @param row The vector to store the new values.
 * @param row_swap The vector of old values.
 */
static void iterate_row(const int width,
                        FP_TYPE *const restrict row,
                        const FP_TYPE *const restrict row_swap)
{
    //We don't need to have a special case for the right edge because there
    //are extra columns beyond the right edge with the needed values.
    for (int j = 2; j < width; j++)
    {
        //Using the appropriate types for the constants helps in the CPU case too.
#ifdef USE_DOUBLE
        row[j] = 0.34 * row_swap[j - 2] + 0.28 * row_swap[j - 1] + 0.2 * row_swap[j] + 0.12 * row_swap[j + 1] + 0.06 * row_swap[j + 2];
#else
        row[j] = 0.34f * row_swap[j - 2] + 0.28f * row_swap[j - 1] + 0.2f * row_swap[j] + 0.12f * row_swap[j + 1] + 0.06f * row_swap[j + 2];
#endif
    }

    return;
}

/**
 * @brief Applies the given number of iterations to a row. Doing each row 
 * seperately makes better use of cache and is about 40% faster than iterating
 * the whole matrix at a time.
 * 
 * @param iterations The number of iterations.
 * @param width The width of the row.
 * @param row The vector to store the finished values. Should be set to the
 * initial conditions.
 * @param row_swap A vector set to the initial conditions. Used to store 
 * intermediary values.
 */
static void apply_iterations_row(const int iterations, const int width,
                                 FP_TYPE *restrict row, FP_TYPE *restrict row_swap)
{
    if (iterations % 2 != 0)
    {
        iterate_row(width, row, row_swap);
    }
    for (int iter = 0; iter < iterations / 2; iter++)
    {
        iterate_row(width, row_swap, row);
        iterate_row(width, row, row_swap);
    }

    return;
}

/**
 * @brief Applies the given number of iterations to the matrix, one row at a 
 * time.
 * 
 * @param height The height of the matrix.
 * @param width The width (excluding the extra colums) of the matrix.
 * @param matrix The matrix to store the result. Should be have height X (width + 2)
 * entries allocated.
 * @param matrix_swap A matrix with the same dimensions used to store 
 * intermediary values.
 */
void apply_iterations(const int iterations,
                      const int height, const int width,
                      FP_TYPE *restrict matrix, FP_TYPE *restrict matrix_swap)
{
    //We assume there are two extra columns to simplify the wrapping around.
    const int stride = width + 2;

    /* Set initial conditions */

    set_initial_conditions(height, width, matrix);
    set_initial_conditions(height, width, matrix_swap);

    /* Loop over rows and apply iterations */

    //This could be trivially parallelised.
    // #pragma omp parallel for
    for (int i = 0; i < height; i++)
    {
        FP_TYPE *const row = matrix + i * stride;           /* A pointer to the row in matrix */
        FP_TYPE *const row_swap = matrix_swap + i * stride; /* A pointer to the row in matrix_swap */

        apply_iterations_row(iterations, width, row, row_swap);
    }

    return;
}
