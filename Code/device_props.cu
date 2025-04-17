/**
 * @file select_device.cu
 * @author Sam McKeown (mckeowsa@tcd.ie)
 * @brief Provides functions for querying and selecting a GPU.
 * @version 1.0
 * @date 2022-04-04
 * 
 * 
 * 
 */

#include "device_props.h"

/**
 * @brief A simple kernel which does almost no work before exiting.
 */
__global__ void dummy_kernel(void)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id >= 0)
    {
        global_id += 1;
    }
}

/**
 * @brief Returns the index of the current device.
 * 
 * @return int The index of the current device. Negative on error.
 */
int current_device(void)
{
    int index = -1;
    if(cudaGetDevice(&index) != cudaSuccess)
    {
        return -1;
    }

    return index;
}

/**
 * @brief Gives the compute capability of the given device, in the same format 
 * as __CUDA_ARCH__.
 * 
 * @param device_index The device to check.
 * @return int The compute capability. Negative on error.
 */
int compute_capability(const int device_index)
{
    int major_cc = -1;
    if (cudaDeviceGetAttribute(&major_cc, cudaDevAttrComputeCapabilityMajor, device_index) != cudaSuccess)
    {
        return -1;
    }

    int minor_cc = -1;
    if (cudaDeviceGetAttribute(&minor_cc, cudaDevAttrComputeCapabilityMinor, device_index) != cudaSuccess)
    {
        return -1;
    }

    return 100 * major_cc + 10 * minor_cc;
}

/**
 * @brief Gives the compute capability of the current device, in the same format
 *  as __CUDA_ARCH__.
 * 
 * @return int The compute capability. Negative on error.
 */
int current_compute_capability(void)
{
    return compute_capability(current_device());
}

/**
 * @brief Gives the given card's maximum allowed x dimemsion for grids.
 * 
 * @param device_index The device to check.
 * @return int The maximum x dimension. Negative on error.
 */
int max_grid_dim_x(const int device_index)
{
    int dim= -1;

    if (cudaDeviceGetAttribute(&dim, cudaDevAttrMaxGridDimX, device_index) != cudaSuccess)
    {
        return -1;
    }

    return dim;
}

/**
 * @brief Gives the current card's maximum allowed x dimemsion for grids.
 * 
 * @return int The maximum x dimension. Negative on error.
 */
int current_max_grid_dim_x(void)
{
    return max_grid_dim_x(current_device());
}

/**
 * @brief Gives the maximum usable shared memory per block on the given device.
 * 
 * @param device_index The device to check.
 * @return int The maximum usable shared memory, in bytes. Negative on error.
 */
int max_shared_per_block(const int device_index)
{
    int max = -1;

    if (cudaDeviceGetAttribute(&max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_index) != cudaSuccess)
    {
        return -1;
    }

    return max;
}

/**
 * @brief Gives the maximum usable shared memory per block on the current 
 * device.
 * 
 * @return int The maximum usable shared memory, in bytes. Negative on error.
 */
int current_max_shared_per_block(void)
{
    return max_shared_per_block(current_device());
}

/**
 * @brief Gives the number of SMs on the given card.
 * 
 * @param device_index The device to check.
 * @return int The number of SMs. Negative on error.
 */
int multiprocessor_count(const int device_index)
{
    int sm_count = -1;
    if (cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_index) != cudaSuccess)
    {
        return -1;
    }

    return sm_count;
}

/**
 * @brief Gives the number of SMs on the current card.
 * 
 * @return int The number of SMs. Negative on error.
 */
int current_multiprocessor_count(void)
{
    return multiprocessor_count(current_device());
}

/**
 * @brief Gives the number of CUDA cores on the given card. This is largely 
 * taken from the example code.
 * 
 * @param device_index The device to check.
 * @return int The number of CUDA cores. Negative on error.
 */
int cuda_core_count(const int device_index)
{
    int sm_count = multiprocessor_count(device_index);
    if(sm_count < 0)
    {
        return -1;
    }

    int major_cc = -1;
    if (cudaDeviceGetAttribute(&major_cc, cudaDevAttrComputeCapabilityMajor, device_index) != cudaSuccess)
    {
        return -1;
    }

    int minor_cc = -1;
    if (cudaDeviceGetAttribute(&minor_cc, cudaDevAttrComputeCapabilityMinor, device_index) != cudaSuccess)
    {
        return -1;
    }

    int cc_per_sm = -1;
    switch (major_cc)
    {
    case 1: // Tesla / T10
        cc_per_sm = 8;
        break;
    case 2: // Fermi
        cc_per_sm = 32;
        break;
    case 3: // Kepler
        cc_per_sm = 192;
        break;
    case 5: // Maxwell
        cc_per_sm = 128;
        break;
    case 6: // Pascal
        switch (minor_cc)
        {
        case 0: // GP100, 64 cuda cores per SM - 7.0 should be prefered over 7.1
            cc_per_sm = 64;
            break;
        case 1: // GP102, GP104, GP106, GP107, 128 cuda cores per SM
            cc_per_sm = 128;
            break;
        default: // Unknown - 6.2 is the GP10B on Jetson TX2, DRIVE PX 2
            cc_per_sm = -1;
            break;
        }
        break;
    case 7: // Volta is 7.0 and 7.2, 64 cuda cores per SM
        switch (minor_cc)
        {
        case 0:
            cc_per_sm = 64;
            break;
        case 2:
            cc_per_sm = 64;
            break;
        case 5: // Turing is 7.5 - also has 64 cuda cores per SM
            cc_per_sm = 64;
            break;
        default: // Unknown?
            cc_per_sm = -1;
            break;
        }
        break;
    case 8: // Ampere, 64 cuda cores per SM
        cc_per_sm = 64;
        break;
    default: // Unknown
        cc_per_sm = -1;
        break;
    }

    return sm_count * cc_per_sm;
}

/**
 * @brief Gives the number of CUDA cores on the current card.
 * 
 * @return int The number of CUDA cores. Negative on error.
 */
int current_cuda_core_count(void)
{
    return cuda_core_count(current_device());
}

/**
 * @brief A function which returns the device index of the device with the most
 * CUDA cores. This is mostly taken from the example code.
 * 
 * @return int The index of the device with the most cores. Negative on error.
 */
int best_device_index(void)
{
    /* Initialise variables */

    int device_count = -1;

    if (cudaGetDeviceCount(&device_count) != cudaSuccess)
    {
        return -1;
    }

    int best_index = -1;
    int best_count = -1;

    /* Loop over devices and calculate core count */

    for (int i = 0; i < device_count; i++)
    {       
        const int cc_count = cuda_core_count(i);

        //I changed the logic here slightly as I felt there was a mistake in
        //the sample code. We multiply the number of processors by the cores
        //per processor, and select the device with the largest value.
        if (cc_count > best_count)
        {
            best_count = cc_count;
            best_index = i;
        }
    }

    return best_index;
}

/**
 * @brief Selects the device with the given index.
 * 
 * @param device_index The device to set.
 * @return int 0 on success or negative on error.
 */
int set_device(const int device_index)
{
    if (cudaSetDevice(device_index) != cudaSuccess)
    {
        return -1;
    }

    return 0;
}

/**
 * @brief A dummy function to "warm up" the GPU and allow for accurate timing
 * of subsequent functions.
 * 
 * @return int Returns 0.
 */
int dummy_function_gpu(void)
{
    dummy_kernel<<<1, 1>>>();

    return 0;
}
