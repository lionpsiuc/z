/**
 * @file fp_type.h
 * @author Sam McKeown (mckeowsa@tcd.ie)
 * @brief A header file to facilitate switching between floats and doubles at
 * compile time.
 * @version 1.0
 * @date 2022-03-05
 * 
 * 
 * 
 */

#ifndef FP_TYPE_H
#define FP_TYPE_H

#include <math.h>

/* Switches between floats and doubles at compile time */

#ifdef USE_DOUBLE
typedef double FP_TYPE;
#define ABS(X) fabs(X)
#else
typedef float FP_TYPE;
#define ABS(X) fabsf(X)
#endif

#endif