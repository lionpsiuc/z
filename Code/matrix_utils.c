/**
 * @file matrix_utils.c
 * @author Sam McKeown (mckeowsa@tcd.ie)
 * @brief Implements some functions for working with matrices.
 * @version 1.0
 * @date 2022-04-04
 * 
 * 
 * 
 */

#include "matrix_utils.h"

/**
 * @brief Gives simple formatted output for the given matrix.
 * 
 * @param height The height of the matrix.
 * @param width The width of the matrix.
 * @param stride The gap between the starts of subsequent rows.
 * @param A The matrix to print.
 */
void print_matrix(const int height, const int width,
                  const int stride, FP_TYPE *const A)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            printf("%8.6f   ", A[i * stride + j]);
            // printf("%4.2e   ", A[i * stride + j]);
        }
        printf("\n");
    }
}

/**
 * @brief Counts the number of differences which are above the given threshold.
 * 
 * @param tolerance The threshold for counting discrepancies.
 * @param height The height of the matrices.
 * @param width The width of the matrices.
 * @param stride_A The gap between rows of A.
 * @param A The first matrix to compare.
 * @param stride_B The gap between rows of B.
 * @param B The second matrix to compare.
 * @return int The number of disrepancies between A and B.
 */
int count_discrepancies(const double tolerance,
                        int height, const int width,
                        const int stride_A, const FP_TYPE *const A,
                        const int stride_B, const FP_TYPE *const B)
{
    int count = 0;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            //I used doubles since this is a diagnostic function
            //and not performance sensitive.
            if (fabs((double)A[stride_A * i + j] - B[stride_B * i + j]) >= tolerance)
            {
                count++;
            }
        }
    }

    return count;
}

/**
 * @brief Computes the average absolute difference between elements of A and B.
 * 
 * @param height The height of the matrices.
 * @param width The width of the matrices.
 * @param stride_A The gap between rows of A.
 * @param A The first matrix to compare.
 * @param stride_B The gap between rows of B.
 * @param B The second matrix to compare.
 * @return double The average difference between elements of A and B.
 */
double L1_diff(const int height, const int width,
               const int stride_A, const FP_TYPE *const A,
               const int stride_B, const FP_TYPE *const B)
{
    //I used doubles since this is a diagnostic function
    //and not performance sensitive.
    double diff = 0.0;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            diff += fabs((double)A[stride_A * i + j] - B[stride_B * i + j]);
        }
    }

    const double unit_area = 1.0 / (height * width);

    return diff * unit_area;
}
