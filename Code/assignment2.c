#include <stdio.h>
#include <stdlib.h>

#include "fp_type.h"

#include "cl_arguments.h"
#include "matrix_utils.h"
#include "timing.h"

#include "average.h"
#include "iteration.h"

#include "average_gpu.h"
#include "iteration_gpu.h"

#include "device_props.h"

int main(int argc, char* argv[]) {
  /* Parse arguments */

  const CL_arguments arguments = parse_arguments(argc, argv);

  const int height = arguments.height;
  const int width  = arguments.width;
  const int stride = width + 2;

  const int iterations = arguments.iterations;

  const double tolerance = arguments.tolerance;

  const int device_index = (arguments.device_index >= 0)
                               ? arguments.device_index
                               : best_device_index();

  const int block_size = arguments.block_size;

  const bool skip_cpu    = arguments.skip_cpu;
  const bool do_average  = arguments.do_average;
  const bool show_timing = arguments.show_timing;
  const bool show_values = arguments.show_values;
  const bool show_csv    = arguments.show_csv;

  /* Select CUDA device */

  if (device_index < 0) {
    fprintf(stderr, "Couldn't find CUDA device. Exiting.\n");
    exit(EXIT_FAILURE);
  }
  if (set_device(device_index) != 0) {
    fprintf(stderr, "Failed to set CUDA device. Exiting.\n");
    exit(EXIT_FAILURE);
  }

  /* Allocate memory on the host */

  FP_TYPE* const matrix      = calloc(height * stride, sizeof(FP_TYPE));
  FP_TYPE* const matrix_swap = calloc(height * stride, sizeof(FP_TYPE));
  FP_TYPE* const averages    = calloc(height, sizeof(FP_TYPE));

  FP_TYPE* const matrix_cuda   = calloc(height * stride, sizeof(FP_TYPE));
  FP_TYPE* const averages_cuda = calloc(height, sizeof(FP_TYPE));

  if (matrix == NULL || matrix_swap == NULL || averages == NULL ||
      matrix_cuda == NULL || averages_cuda == NULL) {
    fprintf(stderr, "Failed to allocate memory on host. Exiting.\n");
    exit(EXIT_FAILURE);
  }

  /* Set up CPU timing */

  double cpu_timings[2] = {0};
  int    timings_index  = 0;
  double start_time     = get_current_time();

  /* Do iterations on the CPU */

  if (!skip_cpu) {
    apply_iterations(iterations, height, width, matrix, matrix_swap);
    cpu_timings[timings_index] = get_duration(&start_time);
    timings_index++;
  }

  /* Set up GPU timing and parameters */

  float iteration_timings[5] = {0}; /* Stores times measured with CUDA events */

  // We add 2 so that we can pass a multiple of 32 as an argument and
  // still have a suitable block size.
  const int iteration_block_size =
      (block_size + 2 > MAX_BLOCK_SIZE)
          ? MAX_BLOCK_SIZE
          : block_size + 2; /* The block size to use */

  // Warming up the GPU will give more representative CPU-side timing
  dummy_function_gpu();

  /* Do iterations on the GPU */

  get_duration(&start_time);
  apply_iterations_gpu(iterations, height, width, matrix_cuda,
                       iteration_block_size, iteration_timings);
  const double iteration_total_time =
      get_duration(&start_time); /* Stores the time as measured by the CPU */

  /* Calculate differences in results */

  const double matrix_diff = L1_diff(height, width, stride, matrix, width,
                                     matrix_cuda); /* The average difference */

  const int matrix_disc =
      count_discrepancies(tolerance, height, width, stride, matrix, width,
                          matrix_cuda); /* The count of large errors */

  /* Write parameters to output */

  printf("\nMatrix:\t\t%d x %d (x %d iterations)\n", height, width, iterations);
  printf("Precision:\tFP%lu\n", 8 * sizeof(FP_TYPE));
  printf("Tolerance:\t%4.2e\n", tolerance);
  printf("Device:\t\t%d\n", device_index);
  printf("Block size:\t%d (+%d)\n\n", block_size,
         iteration_block_size - block_size);

  /* Write output for iterations */

  printf("ITERATIONS\n\n");
  printf("\tL1 Difference:\t\t%4.2e\n", matrix_diff);
  printf("\tDiffs over tol.:\t%d\n", matrix_disc);
  if (show_timing) {
    printf("\n\t-------------------------------------------------\n\n");

    if (!skip_cpu) {
      printf("\tSpeedup (overall):\t%4.2lf\n",
             cpu_timings[0] / iteration_total_time);
      printf("\tSpeedup (computation):\t%4.2lf\n",
             cpu_timings[0] / iteration_timings[3]);
      printf("\n\t-------------------------------------------------\n\n");
    }

    printf("\tTIMINGS\t\tCPU\t\tGPU\n\n");
    printf("\tTotal time\t%4.2e\t%4.2e\n\n", cpu_timings[0],
           iteration_total_time);
    printf("\tSetup\t\t\t\t%4.2e (%05.2lf%%)\n", iteration_timings[0],
           100 * iteration_timings[0] / iteration_total_time);
    printf("\tAllocation\t\t\t%4.2e (%05.2lf%%)\n", iteration_timings[1],
           100 * iteration_timings[1] / iteration_total_time);
    printf("\tTransfer to\t\t\t%4.2e (%05.2lf%%)\n", iteration_timings[2],
           100 * iteration_timings[2] / iteration_total_time);
    printf("\tComputation\t\t\t%4.2e (%05.2lf%%)\n", iteration_timings[3],
           100 * iteration_timings[3] / iteration_total_time);
    printf("\tTransfer from\t\t\t%4.2e (%05.2lf%%)\n", iteration_timings[4],
           100 * iteration_timings[4] / iteration_total_time);
  }

  /* Write results for matrix */

  if (show_values) {
    if (!skip_cpu) {
      printf("\nCPU result:\n");
      print_matrix(height, width, stride, matrix);
    }
    printf("\nGPU result:\n");
    print_matrix(height, width, width, matrix_cuda);
    printf("\n");
  }

  /* Do the averaging */

  if (do_average) {
    FP_TYPE overall_average = 0.0f; /* Will store the average across all rows */
    if (!skip_cpu) {
      if (do_average) {
        /* Do averaging on the CPU */

        get_duration(&start_time);
        average_rows(height, width, stride, matrix, averages);
        cpu_timings[timings_index] = get_duration(&start_time);
        timings_index++;

        /* Get the overall average on the CPU */

        average_rows(1, height, height, averages, &overall_average);
      }
    }

    /* Set up GPU timing */

    float average_timings[5] = {0}; /* Stores times measured with CUDA events */

    /* Do averaging on the GPU */

    get_duration(&start_time);
    average_rows_gpu(height, width, width, matrix_cuda, averages_cuda,
                     block_size, average_timings);
    const double average_total_time =
        get_duration(&start_time); /* Stores the time as measured by the CPU */

    /* Get the overall average on the GPU */

    FP_TYPE overall_average_cuda =
        0.0f; /* Will store the average across all rows */
    average_rows_gpu(1, height, height, averages_cuda, &overall_average_cuda,
                     block_size, NULL);

    /* Calculate differences in results */

    const double average_diff = L1_diff(
        height, 1, 1, averages, 1, averages_cuda); /* The average difference */

    const int average_disc =
        count_discrepancies(tolerance, height, 1, 1, averages, 1,
                            averages_cuda); /* The count of large errors */

    /* Write output for the averaging */

    printf("\n\nAVERAGES\n\n");
    printf("\tL1 Difference:\t\t%4.2e\n", average_diff);
    printf("\tDiffs over tol.:\t%d\n", average_disc);
    printf("\n\t-------------------------------------------------\n\n");
    printf("\tOverall (CPU):\t\t%4.2e\n", overall_average);
    printf("\tOverall (GPU):\t\t%4.2e\n", overall_average_cuda);
    if (show_timing) {
      printf("\n\t-------------------------------------------------\n\n");

      if (!skip_cpu) {
        printf("\tSpeedup (overall):\t%4.2lf\n",
               cpu_timings[1] / average_total_time);
        printf("\tSpeedup (computation):\t%4.2lf\n",
               cpu_timings[1] / average_timings[3]);
        printf("\n\t-------------------------------------------------\n\n");
      }

      printf("\tTIMINGS\t\tCPU\t\tGPU\n\n");
      printf("\tTotal time\t%4.2e\t%4.2e\n\n", cpu_timings[1],
             average_total_time);
      printf("\tSetup\t\t\t\t%4.2e (%05.2lf%%)\n", average_timings[0],
             100 * average_timings[0] / average_total_time);
      printf("\tAllocation\t\t\t%4.2e (%05.2lf%%)\n", average_timings[1],
             100 * average_timings[1] / average_total_time);
      printf("\tTransfer to\t\t\t%4.2e (%05.2lf%%)\n", average_timings[2],
             100 * average_timings[2] / average_total_time);
      printf("\tComputation\t\t\t%4.2e (%05.2lf%%)\n", average_timings[3],
             100 * average_timings[3] / average_total_time);
      printf("\tTransfer from\t\t\t%4.2e (%05.2lf%%)\n", average_timings[4],
             100 * average_timings[4] / average_total_time);
    }

    /* Write results for averages */

    if (show_values) {
      if (!skip_cpu) {
        printf("\nCPU result:\n");
        print_matrix(height, 1, 1, averages);
      }
      printf("\nGPU result:\n");
      print_matrix(height, 1, 1, averages_cuda);
      printf("\n");
    }

    /* Write the CSV output */

    if (show_csv) {
      fprintf(stderr, "%d,%d,%d,%lu,", height, width, iterations,
              8 * sizeof(FP_TYPE));
      fprintf(stderr, "%d,%d,%d,", skip_cpu, device_index, block_size);

      fprintf(stderr, "%e,%e,", cpu_timings[0], iteration_total_time);
      for (int i = 0; i < 5; i++) {
        fprintf(stderr, "%e,", iteration_timings[i]);
      }
      fprintf(stderr, "%e,%d,", matrix_diff, matrix_disc);

      fprintf(stderr, "%e,%e,", cpu_timings[1], average_total_time);
      for (int i = 0; i < 5; i++) {
        fprintf(stderr, "%e,", average_timings[i]);
      }
      fprintf(stderr, "%e,%d,", average_diff, average_disc);
      fprintf(stderr, "%e,%e\n", overall_average, overall_average_cuda);
    }
  }

  /* Tidy up and exit */

  free(matrix);
  free(matrix_swap);
  free(averages);

  free(matrix_cuda);
  free(averages_cuda);

  return EXIT_SUCCESS;
}
