#ifndef _UTILS_H_
#define _UTILS_H_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "cnpy.h"

extern cnpy::NpyArray read_velocity_field_data( std::string file_path_name, int* n_elements);
extern void define_xs_or_ys(float* xs, float dx, float x0, int gsize);
extern void save_master_Coos_to_file(std::string op_FnamePfx, int num_actions, 
    thrust::host_vector<long long int> &H_master_cooS1, 
    thrust::host_vector<long long int> &H_master_cooS2, 
    thrust::host_vector<float> &H_master_cooVal,
    thrust::host_vector<float> &H_master_R,
    thrust::host_vector<long long int>* H_Aarr_of_cooS1,
    thrust::host_vector<long long int>* H_Aarr_of_cooS2,
    thrust::host_vector<float>* H_Aarr_of_cooProb,
    thrust::host_vector<float>* H_Aarr_of_Rs,
    thrust::host_vector<float> &prob_params,
    long long int* DP_relv_params,
    unsigned long int num_DP_params);


// template<typename dType> template not working for thrust vectors
extern void print_device_vector(thrust::device_vector<long long int> &array, int start_id, int end_id, std::string array_name, std::string end, int method);
extern void make_dir(std::string dir_name);
extern void populate_ac_angles(float* ac_angles, int num_ac_angles);
extern void populate_ac_speeds(float* ac_speeds, int num_ac_speeds, float Fmax);
extern void populate_actions(float** H_actions, int num_ac_speeds, int num_ac_angles, float Fmax);

#endif