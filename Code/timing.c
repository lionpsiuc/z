/**
 * @file timing.c
 * @author Sam McKeown (mckeowsa@tcd.ie)
 * @brief Implements functions for timing on the CPU.
 * @version 1.0
 * @date 2022-04-04
 * 
 * 
 * 
 */

#include "timing.h"

/**
 * @brief Returns the current time in seconds as a double.
 */
double get_current_time(void)
{
    struct timeval current_time;
    gettimeofday(&current_time, NULL);

    return (double)(current_time.tv_sec + current_time.tv_usec * 1e-6);
}

/**
 * @brief Returns the difference between the given time and the current time,
 * then resets the given time.
 * 
 * @param time The start time. Will be reset.
 * @return double The duration since time.
 */
double get_duration(double *const time)
{
    const double diff = get_current_time() - *time;
    *time = get_current_time();

    return diff;
}