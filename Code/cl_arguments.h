/**
 * @file cl_arguments.h
 * @author Sam McKeown (mckeowsa@tcd.ie)
 * @brief A header providing a struct and function for argument parsing.
 * @version 1.0
 * @date 2022-03-05
 *
 *
 *
 */

#ifndef CL_ARGUMENTS_OTSGTO
#define CL_ARGUMENTS_OTSGTO

// We'll use getopts so we should define this if it isn't already
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>

#define MAX_BLOCK_SIZE 1024

/**
 * @brief A struct for wrapping together everything which can be specified at
 * the command line. The defaults are included in the definition of
 * default_arguments().
 *
 */
typedef struct {
  int height; /* The height of the matrix */
  int width;  /* The width of the matrix */

  int iterations; /* The number of iterations */

  double tolerance; /* The threshold for flagging discrepancies */

  int device_index; /* The index of the device to use */
  int block_size;   /* The block size to use */

  bool skip_cpu;    /* Whether to skip the CPU computation */
  bool do_average;  /* Whether to compute average */
  bool show_timing; /* Whether to display times */
  bool show_values; /* Whether to display the matrix values */
  bool show_csv;    /* Whether to give csv output */
} CL_arguments;

CL_arguments parse_arguments(const int argc, char* const argv[]);

#endif
