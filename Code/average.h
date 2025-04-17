#ifndef MATRIX_REDUCE_H
#define MATRIX_REDUCE_H

#include "fp_type.h"

#define VEC_WIDTH 8 /* The vector width to use in the average */

int average_rows(const int height, const int width, const int stride, const FP_TYPE *const matrix, FP_TYPE *const averages);

#endif
