#ifndef _EXTRACT_FIELD_H
#define _EXTRACT_FIELD_H


extern __device__ void extract_velocity(int32_t* posids, long long int sp_id, long long int ncells, float* vx, float* vy,
                                 int32_t T, float* all_u_mat, float* all_v_mat, float* all_ui_mat, 
                                 float* all_vi_mat, float* all_Yi, float* params);
  

#endif