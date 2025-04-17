/**
 * @file average.c
 * @author Sam McKeown (mckeowsa@tcd.ie)
 * @brief Implements a function for averaging rows on the CPU.
 * @version 1.0
 * @date 2022-04-04
 * 
 * 
 * 
 */

#include "average.h"

/**
 * @brief Averages the rows of the given matrix. This is almost identical to the
 * sum_rows() function from the first assignment.
 * 
 * @param height The height of the matrix.
 * @param width The width of the matrix.
 * @param stride The gap between the start of subsequent rows.
 * @param matrix The matrix to average.
 * @param averages A vector to store the results.
 * @return int Returns 0.
 */
int average_rows(const int height, const int width,
                 const int stride, const FP_TYPE *const matrix,
                 FP_TYPE *const averages)
{
    /* Initialise results */
    
    for (int i = 0; i < height; i++)
    {
        averages[i] = 0.0f;
    }

    /* Vectorised sums */

    const int vec_iterations = height / VEC_WIDTH;
    for (int i = 0; i < vec_iterations; i++)
    {
        for (int j = 0; j < width; j++)
        {
            for (int step = 0; step < VEC_WIDTH; step++)
            {
                averages[VEC_WIDTH * i + step] += matrix[(VEC_WIDTH * i + step) * stride + j];
            }
        }
    }

    /* Tail loop */

    for (int i = VEC_WIDTH * vec_iterations; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            averages[i] += matrix[i * stride + j];
        }
    }

    /* Divide by width */

    //Precomputing the invese doesn't seem to affect performance.
    // const FP_TYPE width_inv = (FP_TYPE)1 / width;
    for (int i = 0; i < height; i++)
    {
        // averages[i] *= width_inv;
        averages[i] /= width;
    }

    return 0;
}
