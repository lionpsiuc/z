#ifndef MATRIX_UTILS_H__
#define MATRIX_UTILS_H__

#include <stdio.h>

#include "fp_type.h"

void print_matrix(const int height, const int width, const int stride, FP_TYPE *const A);
int count_discrepancies(const double tolerance, int height, const int width, const int stride_A, const FP_TYPE *const A, const int stride_B, const FP_TYPE *const B);
double L1_diff(const int height, const int width, const int stride_A, const FP_TYPE *const A, const int stride_B, const FP_TYPE *const B);

#endif // MATRIX_UTILS_H__
