#include "extract_field.h"
#include "get_funcs.h"


__device__ void extract_velocity(int32_t* posids, long long int sp_id, long long int ncells, float* vx, float* vy,
    int32_t T, float* all_u_mat, float* all_v_mat, float* all_ui_mat, 
    float* all_vi_mat, float* all_Yi, float* params){



    int32_t nrzns = params[2];
    int32_t nmodes = params[7];    
    int32_t gsize = params[0];          

    long long int sp_uvi, str_uvi, sp_Yi; //startpoints and strides for accessing all_ui_mat, all_vi_mat and all_Yi
    // int str_Yi;
    float sum_x = 0;
    float sum_y = 0;
    float vx_mean, vy_mean;
    //thread index. also used to access resultant vxrzns[nrzns, gsize, gsize]
    int32_t idx = get_thread_idx();
    //rzn index to identify which of the 5k rzn it is. used to access all_Yi.
    int32_t rzn_id = get_rzn_id() ;
    //mean_id is the index used to access the flattened all_u_mat[t,i,j].
    long long int mean_id = state1D_from_thread(T, sp_id, ncells);
    //to access all_ui_mat and all_vi_mat
    //str_uvi = gridDim.x * gridDim.y;
    // sp_uvi = (T * nmodes * str_uvi) + (gridDim.x * blockIdx.y) + (blockIdx.x);
    str_uvi = gsize*gsize*1LL;
    sp_uvi = (T * nmodes * str_uvi) + (gsize * posids[0]) + (posids[1]);

    // to access all_Yi
    sp_Yi = (T * nrzns * nmodes * 1LL) + (rzn_id * nmodes);
    vx_mean = all_u_mat[mean_id];
    for(int i = 0; i < nmodes; i++)
    {
    sum_x += all_ui_mat[sp_uvi + (i*str_uvi)]*all_Yi[sp_Yi + i];
    }
    vy_mean = all_v_mat[mean_id];
    for(int i = 0; i < nmodes; i++)
    {
    sum_y += all_vi_mat[sp_uvi + (i*str_uvi)]*all_Yi[sp_Yi + i];
    }

    *vx = vx_mean + sum_x;
    *vy = vy_mean + sum_y;

    return;
}


__device__ void extract_radiation(long long int sp_id, int32_t T, long long int ncells, 
                                float* all_s_mat, float* rad){
    // for DETERMINISTIC radiation (scalar) field
    // extract radiation (scalar) from scalar matrix 
    
    long long int mean_id = state1D_from_thread(T, sp_id, ncells);
    *rad = all_s_mat[mean_id];

    return;
}



// __device__ void extract_radiation(int32_t* posids, long long int sp_id, long long int ncells, float* rad,
//     int32_t T, float* all_s_mat, float* all_si_mat, float* all_sYi, float* params){

//     // s and rad imply the same thing codewise. s is scalar, rad is radation
//     int32_t s_nrzns = params[18];
//     int32_t s_nmodes = params[19];    
//     int32_t gsize = params[0];          

//     long long int sp_s_i, str_s_i, sp_sYi; //startpoints and strides for accessing all_si_mat, all_vi_mat and all_sYi
//     // int str_Yi;
//     float sum = 0;
//     float s_mean;
//     //thread index. also used to access resultant radrzns[s_nrzns, gsize, gsize]
//     int32_t idx = get_thread_idx();
//     //rzn index to identify which of the 5k rzn it is. used to access all_sYi.
//     int32_t rzn_id = get_rzn_id() ;
//     //mean_id is the index used to access the flattened all_s_mat[t,i,j].
//     long long int mean_id = state1D_from_thread(T, sp_id, ncells);
//     //to access all_si_mat and all_vi_mat
//     //str_s_i = gridDim.x * gridDim.y;
//     // sp_s_i = (T * nmodes * str_s_i) + (gridDim.x * blockIdx.y) + (blockIdx.x);
//     str_s_i = gsize*gsize*1LL;
//     sp_s_i = (T * s_nmodes * str_s_i) + (gsize * posids[0]) + (posids[1]);

//     // to access all_sYi
//     sp_sYi = (T * s_nrzns * s_nmodes * 1LL) + (rzn_id * s_nmodes);
//     s_mean = all_s_mat[mean_id];
//     for(int i = 0; i < s_nmodes; i++)
//     {
//     sum += all_si_mat[sp_s_i + (i*str_s_i)]*all_sYi[sp_sYi + i];
//     }
    
//     *rad = s_mean + sum;

//     return;
// }