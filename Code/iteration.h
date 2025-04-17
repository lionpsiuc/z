#ifndef ITERATION_H__
#define ITERATION_H__

#include "fp_type.h"

void apply_iterations(const int iterations, const int height, const int width, FP_TYPE *restrict matrix, FP_TYPE *restrict matrix_swap);

#endif // ITERATION_H__
