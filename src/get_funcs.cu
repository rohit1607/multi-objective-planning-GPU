#include"get_funcs.h"

__device__ int32_t get_thread_idx(){
    // assigns idx to thread with which it accesses the flattened 3d vxrzns matrix
    // for a given T and a given action. 
    // runs for both 2d and 3d grid
    // TODO: may have to change this considering cache locality
    // here i, j, k refer to a general matrix M[i][j][k]
    int32_t i = threadIdx.x;
    int32_t j = blockIdx.y;
    int32_t k = blockIdx.x;
    int32_t idx = k + (j*gridDim.x)  + (i*gridDim.x*gridDim.y)+ blockIdx.z*blockDim.x*gridDim.x*gridDim.y;
    return idx;
}

__device__ long long int state1D_from_thread(int32_t T, int32_t sp_id, long long int ncells){   
    // j ~ blockIdx.x
    // i ~ blockIdx.y 
    // The above three consitute a spatial state index from i and j of grid
    // last term is for including time index as well.

        // return value when full spatial grid was used
        // return (blockIdx.x + (blockIdx.y*gridDim.x) + (T*gridDim.x*gridDim.y) ); 
    
    // return value for chunks concept
    return sp_id + (T*ncells);
}


__device__ long long int state1D_from_ij(int32_t*  posid, int32_t T, int32_t gsize){
    // posid = {i , j}
    // state id = j + i*dim(i) + T*dim(i)*dim(j)

        // return value when full spatial grid was used
        // return (posid[1] + posid[0]*gridDim.x + (T*gridDim.x*gridDim.y) ) ; 

    // return value for chunks concept
    return (posid[1] + posid[0]*gsize + (T*gsize*gsize) ) ; 

}


__device__ int32_t get_rzn_id(){

    return (blockIdx.z * blockDim.x)  + threadIdx.x;
}

__device__ void get_posids_from_sp_id(long long int sp_id, int gsize, int32_t* posids){

    posids[0] = sp_id/gsize;
    posids[1] = sp_id%gsize;
    return;
}

__device__ long long int get_sp_id(int chunkNum, int chunk_size){

    return (chunkNum*chunk_size)*1LL + blockIdx.x;
}


__device__ bool is_edge_state(int32_t i, int32_t j){
    // n = gsize -1 that is the last index of the domain assuming square domain
    int32_t n = gridDim.x - 1;
    if (i == 0 || i == n || j == 0 || j == n ) 
        return true;
    else 
        return false;
}


__device__ bool is_terminal(int32_t i, int32_t j, float* params){
    // terminal state indices (of UL corner of terminal subgrid if term_subgrid_size>1)
    int32_t i_term = params[8];         
    int32_t j_term = params[9];
    int tsgsize = params[12]; //term_subgrid_size

    if( (i >= i_term && i < i_term + tsgsize)  && (j >= j_term && j < j_term + tsgsize) )
        return true;
    else return false;
}


__device__ bool my_isnan(int s){
    // By IEEE 754 rule, NaN is not equal to NaN
    return s != s;
}


__device__ void get_xypos_from_ij(int32_t i, int32_t j, int32_t gsize ,float* xs, float* ys, float* x, float* y){
    *x = xs[j];
        // *y = ys[gridDim.x - 1 - i];
    *y = ys[gsize - 1 - i];

    return;
}


__device__ float get_angle_in_0_2pi(float theta){
    float f_pi = 3.141592;
    if (theta < 0)
        return theta + (2*f_pi);
    else
        return theta;
}