/**
 * @file cl_arguments.c
 * @author Sam McKeown (mckeowsa@tcd.ie)
 * @brief A file containing functions for parsing command line arguments and
 * defining default values.
 * @version 1.0
 * @date 2022-03-05
 *
 *
 *
 */

#include "cl_arguments.h"

/**
 * @brief Prints the command line usage.
 *
 */
static void print_help() {
  printf("Usage:");
  printf(" ./assignment2");
  printf(" [ -n height ]");
  printf(" [ -m width ]");
  printf(" [ -p iterations ]");
  printf(" [ -e tolerance ]");
  printf(" [ -d device_index ]");
  printf(" [ -b block_size ]");
  printf(" [ -a ]");
  printf(" [ -c ]");
  printf(" [ -t ]");
  printf(" [ -v ]");
  printf(" [ -o ]");
  printf(" [ -h ]");
  printf("\n");
}

/**
 * @brief A conveniece function for reading in variables from strings. As it's
 * intended to tidy up the parsing function, the errors are checked here and the
 * program exited if one occures.
 *
 * @param argument_flag The flag currently being read by getopts.
 * @param variable The variable into which we are to read the input.
 * @param format The format string with which to parse the input.
 */
static void read_argument(const char argument_flag, const void* const variable,
                          const char format[]) {
  if (sscanf(optarg, format, variable) != 1) {
    fprintf(stderr, "Couldn't read argument for -%c. Exiting.\n",
            argument_flag);
    print_help();
    exit(EXIT_FAILURE);
  }
}

/**
 * @brief This produces a CL_arguments struct with everything set to the default
 * values. The defaults can be changed by editing this definition.
 *
 * @return CL_arguments A struct containing the default arguments.
 */
static CL_arguments default_arguments(void) {
  CL_arguments arguments;

  arguments.height = 32;
  arguments.width  = 32;

  arguments.iterations = 10;

  arguments.tolerance = 1e-5;

  arguments.device_index = -1;
  arguments.block_size   = 32;

  arguments.skip_cpu    = false;
  arguments.do_average  = false;
  arguments.show_timing = false;
  arguments.show_values = false;
  arguments.show_csv    = false;

  return arguments;
}

/**
 * @brief Parses the command line arguments and produces a struct containing the
 * values it read along with the defaults for those variables not specified.
 *
 * @param argc The count of arguments.
 * @param argv The vector of argument strings.
 * @return CL_arguments A struct containing the parsed arguments and the default
 * values for any not specified.
 */
CL_arguments parse_arguments(const int argc, char* const argv[]) {
  /* Start with defaults */

  CL_arguments arguments = default_arguments();

  /* Parse arguments */

  const char argument_list[] = "n:m:p:e:d:b:actvoh";
  char       argument_flag;

  while ((argument_flag = getopt(argc, argv, argument_list)) != -1) {
    switch (argument_flag) {
      case 'n': read_argument(argument_flag, &(arguments.height), "%d"); break;
      case 'm': read_argument(argument_flag, &(arguments.width), "%d"); break;
      case 'p':
        read_argument(argument_flag, &(arguments.iterations), "%d");
        break;
      case 'e':
        read_argument(argument_flag, &(arguments.tolerance), "%lf");
        break;
      case 'd':
        read_argument(argument_flag, &(arguments.device_index), "%d");
        break;
      case 'b':
        read_argument(argument_flag, &(arguments.block_size), "%d");
        break;
      case 'a': arguments.do_average = true; break;
      case 'c': arguments.skip_cpu = true; break;
      case 't': arguments.show_timing = true; break;
      case 'v': arguments.show_values = true; break;
      case 'o': arguments.show_csv = true; break;
      case 'h':
        print_help();
        exit(EXIT_SUCCESS);
        break;
      default:
        print_help();
        exit(EXIT_FAILURE);
        break;
    }
  }

  /* Sanity checking */

  if (arguments.width < 5) {
    fprintf(stderr, "Matrix must be at least 5 wide. Exiting.\n");
    exit(EXIT_FAILURE);
  }
  if (arguments.height <= 0) {
    fprintf(stderr, "Matrix must have positive height. Exiting.\n");
    exit(EXIT_FAILURE);
  }
  if (arguments.iterations < 0) {
    fprintf(stderr, "Number of iterations must be non-negative. Exiting.\n");
    exit(EXIT_FAILURE);
  }
  if (arguments.block_size <= 0) {
    fprintf(stderr, "Block size must be positive. Exiting.\n");
    exit(EXIT_FAILURE);
  }
  if (arguments.block_size > MAX_BLOCK_SIZE) {
    fprintf(stderr, "The block size must not exceed %d. Exiting.\n",
            MAX_BLOCK_SIZE);
    exit(EXIT_FAILURE);
  }

  return arguments;
}
