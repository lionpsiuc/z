#ifndef SELECT_DEVICE_H
#define SELECT_DEVICE_H

#ifdef __cplusplus
extern "C"
{
#endif
    int current_device(void);

    int compute_capability(const int device_index);
    int current_compute_capability(void);

    int max_grid_dim_x(const int device_index);
    int current_max_grid_dim_x(void);

    int max_shared_per_block(const int device_index);
    int current_max_shared_per_block(void);

    int multiprocessor_count(const int device_index);
    int current_multiprocessor_count(void);

    int cuda_core_count(const int device_index);
    int current_cuda_core_count(void);

    int best_device_index(void);

    int set_device(const int device_index);
    
    int dummy_function_gpu(void);
#ifdef __cplusplus
}
#endif

#endif
