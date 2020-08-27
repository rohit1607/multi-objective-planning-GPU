#ifndef _GET_FUNCS_H
#define _GET_FUNCS_H



extern __device__ int32_t get_thread_idx();


extern __device__ long long int state1D_from_thread(int32_t T, int32_t sp_id, long long int ncells);


extern __device__ long long int state1D_from_ij(int32_t*  posid, int32_t T, int32_t gsize);


extern __device__ int32_t get_rzn_id();

extern __device__ void get_posids_from_sp_id(long long int sp_id, int gsize, int32_t* posids);

extern __device__ long long int get_sp_id(int chunkNum, int chunk_size);


extern __device__ bool is_edge_state(int32_t i, int32_t j);


extern __device__ bool is_terminal(int32_t i, int32_t j, float* params);


extern __device__ bool my_isnan(int s);


extern __device__ void get_xypos_from_ij(int32_t i, int32_t j, int32_t gsize ,float* xs, float* ys, float* x, float* y);


extern __device__ float get_angle_in_0_2pi(float theta);


#endif