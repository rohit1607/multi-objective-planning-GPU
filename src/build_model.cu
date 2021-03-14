#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/unique.h>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include "cnpy.h"
#include <cmath>
#include <stdlib.h>
#include <fstream>
#include <chrono>
using namespace std::chrono;
#include <iostream>


long long int ncells;


/*
------  Declarations of utility functions from utils.h -------
*/
cnpy::NpyArray read_velocity_field_data( std::string file_path_name, int* n_elements);
void define_xs_or_ys(float* xs, float dx, float x0, int gsize);
void save_master_Coos_to_file(std::string op_FnamePfx, int num_actions, 
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
void print_device_vector(thrust::device_vector<long long int> &array, int start_id, int end_id, std::string array_name, std::string end, int method);
void make_dir(std::string dir_name);
void populate_ac_angles(float* ac_angles, int num_ac_angles);
void populate_ac_speeds(float* ac_speeds, int num_ac_speeds, float Fmax);
void populate_actions(float** H_actions, int num_ac_speeds, int num_ac_angles, float Fmax);




/*
                    ----- Device functions -----
*/

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

__device__ long long int state1D_from_spid(int32_t T, int32_t sp_id, long long int ncells){   
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
    return (posid[1] + posid[0]*gsize + (T*gsize*gsize)*1LL ) ; 

}


__device__ int32_t get_rzn_id(){
    return (blockIdx.z * blockDim.x)  + threadIdx.x;
}

__device__ void get_posids_from_sp_id(long long int sp_id, int gsize, int32_t* posids){

    posids[0] = sp_id/gsize;
    posids[1] = sp_id%gsize;
    return;
}

__device__ long long int get_sp_id(){
    // sp_id: 1d spatial id ranging from 0 to ncells
    int i = blockIdx.y;
    int j = blockIdx.x;
    long long int sp_id = j + (i*gridDim.x)*1LL;
    return sp_id;
}


__device__ void get_posids_relS2_0(int32_t m, int32_t* posids_S1, int32_t* posids_relS2_0){
    // m*m is size of neighbour grid
    // returns i,j index of upper left corner of neighbour grid
    int32_t i1 = posids_S1[0];
    int32_t j1 = posids_S1[1];   
    int32_t del = (m-1)/2;
    posids_relS2_0[0] =  i1 - del;
    posids_relS2_0[1] =  j1 - del;
    return;
}


__device__ long long int get_rel_sp_id2(int32_t m, int32_t* posids_S2, int32_t* posids_relS2_0){
    // returns relative sp_id for S2 in neighbour grid

    int32_t del_i = posids_S2[0] - posids_relS2_0[0]; // i2 - rel_i0
    int32_t del_j = posids_S2[1] - posids_relS2_0[1]; // j2 - rel_j0
    long long int rel_sp_id2;
    // if S2 outstde neighbor grid, assign special value to rel_sp_id2
    // this will help keep count of no. of S2s that lie outside neighb grid
    // ideally there should be 0 such S2s
    if (del_i < m && del_j < m)
        rel_sp_id2 = del_j + (m*del_i);
    else 
        rel_sp_id2 = m*m;               

    return rel_sp_id2;
}



__device__ long long int get_sp_id2_from_rel_sp_id2(int32_t m, int32_t gsize, 
                                    long long int rel_sp_id2, int32_t* posids_relS2_0){
    // returns Sp_id2 from rel_sp_id2
    long long int sp_id2;
    int32_t del_i = rel_sp_id2/m;
    int32_t del_j = rel_sp_id2%m;

    int32_t i2 = del_i + posids_relS2_0[0];
    int32_t j2 = del_j + posids_relS2_0[1];

    sp_id2 = j2 + gsize*i2;

    return sp_id2;
}



__device__ bool is_edge_state(int32_t i, int32_t j){
    // n = gsize -1 that is the last index of the domain assuming square domain
    int32_t n = gridDim.x - 1;
    if (i == 0 || i == n || j == 0 || j == n ) 
        return true;
    else 
        return false;
}


__device__ bool is_in_obstacle(int sp_id, int T, long long int ncells, int* all_mask_mat){
    //returns true if obstacle is present in state T,i,j

    long long int mean_id = state1D_from_spid(T, sp_id, ncells);
    return(all_mask_mat[mean_id] == 1 );

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


__device__ long long int get_sp_id_from_posid(int32_t* posids, int32_t gsize){
    // gives sp_id from posids (i,j)
    return posids[1] + gsize*posids[0]*1LL ;
}


__device__ float get_angle_in_0_2pi(float theta){
    float f_pi = 3.141592;
    if (theta < 0)
        return theta + (2*f_pi);
    else
        return theta;
}






/*
----- move() and reward_functions() moverd to move_and_rewards.h/cu-----
*/

__device__ float calculate_one_step_reward(float ac_speed, float ac_angle, float rad1, float rad2, float* params){

    int method = params[13];
    float Cr = 1;       // coeffecient for radaition term
    float Cf = 1;       // coeffecient for energy consumtion
    float Ct = 0.01;   // small coeffecient for time for to prevent lazy start
    float dt = params[4];

    if (method == 0)    //time
        return -dt;

    else if (method == 1){   //energy1
        return -(Cf*ac_speed*ac_speed + Ct)*dt;
    } 

    else if (method == 2){  //energy2: maximise (collection-consumption)
        return ((Cr*(rad2 + rad1)/2) - (Cf*ac_speed*ac_speed) - Ct)*dt;
    }

    else if (method == 3){
        return ((Cr*(rad2 + rad1)/2)- Ct)*dt;
    }

    else 
        return 0;

}


__device__ void move(float ac_speed, float ac_angle, float vx, float vy, int32_t T, float* xs, float* ys, int32_t* posids, float* params, float* r ){
    int32_t gsize = params[0];
    int32_t n = params[0] - 1;      // gsize - 1
    // int32_t num_actions = params[1];
    // int32_t nrzns = params[2];
    // float F = params[3];
    int32_t nt = params[10];
    float F = ac_speed;
    float dt = params[4];
    float r_outbound = params[5];
    float r_terminal = params[6];
    // int32_t nT = params[10];
    float Dj = fabsf(xs[1] - xs[0]);
    float Di = fabsf(ys[1] - ys[0]);
    int32_t i0 = posids[0];
    int32_t j0 = posids[1];
    float vnetx = F*cosf(ac_angle) + vx;
    float vnety = F*sinf(ac_angle) + vy;
    float x, y;
    get_xypos_from_ij(i0, j0, gsize, xs, ys, &x, &y); // x, y stores centre coords of state i0,j0
    float xnew = x + (vnetx * dt);
    float ynew = y + (vnety * dt);
    // float r_step = 0;
    *r = 0;         // intiilaise r with 0

    if (xnew > xs[n])
        {
            xnew = xs[n];
            *r += r_outbound;
        }
    else if (xnew < xs[0])
        {
            xnew = xs[0];
            *r += r_outbound;
        }
    if (ynew > ys[n])
        {
            ynew =  ys[n];
            *r += r_outbound;
        }
    else if (ynew < ys[0])
        {
            ynew =  ys[0];
            *r += r_outbound;
        }
    int32_t xind, yind;
    float remx = remainderf((xnew - xs[0]), Dj);
    float remy = remainderf(-(ynew - ys[n]), Di);
    xind = ((xnew - xs[0]) - remx)/Dj;
    yind = (-(ynew - ys[n]) - remy)/Di;
    if ((remx >= 0.5 * Dj) && (remy >= 0.5 * Di))
        {
            xind += 1;
            yind += 1;
        }
    else if ((remx >= 0.5 * Dj && remy < 0.5 * Di))
        {
            xind += 1;
        }
    else if ((remx < 0.5 * Dj && remy >= 0.5 * Di))
        {
            yind += 1;
        }
    if (!(my_isnan(xind) || my_isnan(yind)))
        {   
            // update posids
            posids[0] = yind;
            posids[1] = xind;
            if (is_edge_state(posids[0], posids[1]))  //line 110
                {
                    *r += r_outbound;
                }
        
        }

    if (is_terminal(posids[0], posids[1], params))
        {
            *r += r_terminal;
        }
    else{
            //reaching any state in the last timestep which is not terminal is penalised
            if (T == nt-2)
                *r += r_outbound; 
        }

    }




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
    long long int mean_id = state1D_from_spid(T, sp_id, ncells);
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
    
    long long int mean_id = state1D_from_spid(T, sp_id, ncells);
    *rad = all_s_mat[mean_id];

    return;
}


__device__ bool is_within_band(int i, int j, int i1, int j1, int i2, int j2, float* xs, float* ys, int gsize){
    //returns true if i,j are within the band connecticng cells i1,j1 and i2,j2

    if(i1==i2 || j1==j2){
        return true;
    }
    else{
        float x, y, x1, y1, x2, y2;
        float cell_diag = fabsf(xs[1]-xs[0])*1.414213;
        get_xypos_from_ij(i, j, gsize, xs, ys, &x, &y); // x, y stores centre coords of state i0,j0
        get_xypos_from_ij(i1, j1, gsize, xs, ys, &x1, &y1); 
        get_xypos_from_ij(i2, j2, gsize, xs, ys, &x2, &y2);
        float A = (y2-y1)/(x2-x1);
        float B = -1;
        float C = y1 - (A*x1);
        float dist_btw_pt_line = fabsf(A*x + B*y + C)/sqrtf((A*A) + (B*B));
        
        if (dist_btw_pt_line < cell_diag)
            return true;
        else
            return false;
    }
}


__device__ bool goes_through_obstacle(long long int sp_id1, long long int sp_id2, int T, 
                                        long long int ncells, int* D_all_mask_mat, 
                                        float* xs, float* ys, float* params){

    // returns true if the transition involves going through obstacle

    bool possible_collision = false;
    int posid1[2];
    int posid2[2];
    int gsize = params[0];
    long long int sp_id;
    get_posids_from_sp_id(sp_id1, gsize, posid1);
    get_posids_from_sp_id(sp_id2, gsize, posid2);
    int imin = min(posid1[0], posid2[0]);
    int imax = max(posid1[0], posid2[0]);
    int jmin = min(posid1[1], posid2[1]);
    int jmax = max(posid1[1], posid2[1]);
    
    for(int i=imin; i<=imax; i++){
        for(int j=jmin; j<=jmax; j++){
            if(!(i==posid1[0]&&j==posid1[1])){
                sp_id = j + gsize*i*1LL ;
                if ( is_in_obstacle(sp_id, T, ncells, D_all_mask_mat) || is_in_obstacle(sp_id, T+1, ncells, D_all_mask_mat)){
                    if (is_within_band(i, j, posid1[0], posid1[1], posid2[0], posid2[1], xs, ys, gsize) == true){
                        possible_collision = true;
                        return true;
                    }
                }
            }
        }
    }
    
    return possible_collision;
}




//test: changer from float* to float ac_angle
__global__ void transition_calc(float* T_arr, long long int ncells, 
                            float* all_u_mat, float* all_v_mat, float* all_ui_mat, float* all_vi_mat, float* all_Yi,
                            float* D_all_s_mat, int* D_all_mask_mat,
                            float ac_speed, float ac_angle, float* xs, float* ys, float* params, float* sumR_sa, 
                            float* results){
                            // resutls directions- 1: along S2;  2: along S1;    3: along columns towards count
    int32_t gsize = params[0];          // size of grid along 1 direction. ASSUMING square grid.
    int32_t nrzns = params[2]; 
    float r_outbound = params[5];        
    // int32_t is_stationary = params[11];
    int32_t T = (int32_t)T_arr[0];          // current timestep
    int32_t idx = get_thread_idx();
    long long int res_idx;
    float vx, vy, rad1, rad2;
    long long int sp_id = get_sp_id();      //sp_id is space_id. S1%(gsize*gsize)
    long long int sp_id2;
    long long int rel_sp_id2;
    int32_t posids_relS2_0[2];
    int32_t posids_S1[2];
    int32_t m = (int32_t) params[18];
    int32_t Nb = (m*m) + 1;
    float one = 1.0;
    
    if(idx < gridDim.x*gridDim.y*nrzns && sp_id < ncells) //or idx < arr_size
    {
        // int32_t posids[2] = {(int32_t)blockIdx.y, (int32_t)blockIdx.x};    //static declaration of array of size 2 to hold i and j values of S1. 
        int32_t posids[2];    //static declaration of array of size 2 to hold i and j values of S1. 
        get_posids_from_sp_id(sp_id, gsize, posids);
        get_posids_from_sp_id(sp_id, gsize, posids_S1);
        int32_t rzn_id = get_rzn_id();
        
        //  Afer move() these will be overwritten by i and j values of S2
        float r=0;              // to store immediate reward
        float r_step;

        
        extract_velocity(posids, sp_id, ncells, &vx, &vy, T, all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, all_Yi, params);
        extract_radiation(sp_id, T, ncells, D_all_s_mat, &rad1);
        
        // if s1 not terminal
        if (is_terminal(posids[0], posids[1], params) == false){
            // if s1 not in obstacle
            if (is_in_obstacle(sp_id, T, ncells, D_all_mask_mat) == false){

                // moves agent and adds r_outbound and r_terminal to r
                move(ac_speed, ac_angle, vx, vy, T, xs, ys, posids, params, &r);
                sp_id2 = get_sp_id_from_posid(posids, gsize);
                extract_radiation(sp_id2, T+1, ncells, D_all_s_mat, &rad2);
                
                // adds one step-reward based on method. mehthod is available in params
                r_step = calculate_one_step_reward(ac_speed, ac_angle, rad1, rad2, params);
                r += r_step;

                // if S2 is an obstacle cell. then penalise with r_outbound
                // if (is_in_obstacle(sp_id2, T+1, ncells, D_all_mask_mat) == true )
                //     r = r_outbound;
                if (goes_through_obstacle(sp_id, sp_id2, T, ncells, D_all_mask_mat, xs, ys, params) == true)
                    r = r_outbound;
            }
            // if s1 is in obstacle, then no update to posid
            else
                r = r_outbound;
        }
  
        get_posids_relS2_0(m, posids_S1, posids_relS2_0);
        rel_sp_id2 = get_rel_sp_id2(m, posids, posids_relS2_0);
        res_idx = sp_id*Nb + rel_sp_id2;
        float b = atomicAdd(&results[res_idx], one);

        //writing to sumR_sa. this array will later be divided by nrzns, to get the avg
        float a = atomicAdd(&sumR_sa[sp_id], r); 

        __syncthreads();

    }//if ends
    return;
}



__global__ void compute_mean(float* D_master_sumRsa_arr, int size, int nrzns) {
    // computes mean
    int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
    if (tid < size)
        D_master_sumRsa_arr[tid] =  D_master_sumRsa_arr[tid]/nrzns;

    return;
}



__global__ void count_kernel(float* D_master_S2_arr_ip, int nrzns, unsigned long long int* num_uq_s2_ptr) {
    // D_master_S2_arr_ip contains count of relS2s for S1s for a given action
    // This kernel counts no. of nnz elements for a given S1
    // This is needed for getting total nnz to initiliase COO matrix
    // ncells is gridDim,  i.e. we have ncells blocks in grid
    // Nb is blockDim, i.e we have Nb threads in block
    
    int ncells = gridDim.x;  // == ncells == ncells
    int Nb = blockDim.x;
    long long int tid = (blockIdx.x*Nb) + threadIdx.x;
    int idx = blockIdx.x;
    float nnz;
    unsigned long long int one = 1.0;

    if ((tid < ncells*Nb) && (threadIdx.x != Nb-1)){  // tid < Nb*ncells
        if (D_master_S2_arr_ip[tid] != 0){
            nnz = atomicAdd(&num_uq_s2_ptr[idx], one);
        }
    }
    return;
}


__global__ void reduce_kernel(float* D_master_S2_arr_ip, int t, int Nb, int m,
                            long long int ncells, int nrzns, int gsize, 
                            long long int* D_coo_s1_arr, long long int* D_coo_s2_arr, 
                            float* D_coo_cnt_arr, unsigned long long int* num_uq_s2_ptr, unsigned long long int* prSum_num_uq_s2_ptr){

    long long int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
    long long int start_idx = tid*Nb; // to access tid'th threads 0-pos in ip_arr

    long long int n_uqs = num_uq_s2_ptr[tid]; //number of unique S2s for tid'th block
    long long int op_st_id = prSum_num_uq_s2_ptr[tid];   //sum of number of uniqeu S2s uptil tid'th block. to access tid'th thread's 0-pos in op_arr

    long long int ith_nuq = 0; //ranges from 0 to n_uqs , to index number between 0 and n_uqs

    long long int rel_sp_id2;
    long long int sp_id2;
    long long int S2;
    long long int sp_id1 = tid;
    float count; //first if eval will lead to else condition and do  count++ 
    int32_t posids_relS2_0[2];
    int32_t posids_S1[2];


    if (tid < ncells){

        // int32_t s1 = (tid%ncells) + (t*ncells); // TODO:xxdone change this to nbe a function of a arguments: sp_id and t
        long long int s1 = tid + (t*ncells);
        for(long long int i = 0; i< n_uqs; i++)
            D_coo_s1_arr[op_st_id + i] = s1;
        
        get_posids_relS2_0(m, posids_S1, posids_relS2_0);
        for(long long int i = 0; i< Nb-1; i++){
            count = D_master_S2_arr_ip[start_idx + i];
            if (count != 0){
                rel_sp_id2 = i;
                get_posids_from_sp_id(sp_id1, gsize, posids_S1);
                get_posids_relS2_0(m, posids_S1, posids_relS2_0);
                sp_id2 = get_sp_id2_from_rel_sp_id2(m, gsize, 
                    rel_sp_id2, posids_relS2_0);
                S2 = state1D_from_spid(t+1, sp_id2, ncells);
                D_coo_s2_arr[op_st_id + ith_nuq] = S2;         // store old_s2 value in the [.. + ith] position
                D_coo_cnt_arr[op_st_id + ith_nuq] = count/nrzns;   // store prob value in the [.. + ith] position
                ith_nuq++;                                      // increment i
            }
        }

   }
   return;
}


template<typename dType>
void print_array(dType* array, int num_elems,std::string array_name, std::string end){
    std::cout << array_name << std::endl;
    for(int i = 0; i < num_elems; i++)
        std::cout << array[i] << " " << end;
    std::cout << std::endl;
}




std::string get_prob_name(int num_ac_speeds, int num_ac_angles, int i_term, int j_term,
                            int tsg_size){

    std::string s_n_ac_sp = std::to_string(num_ac_speeds);
    std::string s_n_ac_ac = std::to_string(num_ac_angles);
    std::string s_i = std::to_string(i_term);
    std::string s_j = std::to_string(j_term);
    std::string s_tsg = std::to_string(tsg_size);

    std::string name = "a" + s_n_ac_sp + "x" + s_n_ac_ac + "_" 
                        + "i" + s_i + "_" "j" + s_j + "_"
                        + "ref" + s_tsg;

    return name;
}







void build_sparse_transition_model_at_T_at_a(int t, int action_id, int bDimx, thrust::device_vector<float> &D_tdummy, 
                                float* D_all_u_arr, float* D_all_v_arr, float* D_all_ui_arr,
                                float*  D_all_vi_arr, float*  D_all_yi_arr,
                                float* D_all_s_arr, int* D_all_mask_arr,
                                thrust::device_vector<float> &D_params, 
                                thrust::host_vector<float> &H_params,
                                thrust::device_vector<float> &D_xs, 
                                thrust::device_vector<float> &D_ys, 
                                float** H_actions,
                                thrust::host_vector<int32_t> &H_coo_len_per_ac,
                                thrust::host_vector<long long int>* H_Aarr_of_cooS1,
                                thrust::host_vector<long long int>* H_Aarr_of_cooS2,
                                thrust::host_vector<float>* H_Aarr_of_cooProb,
                                thrust::host_vector<float>* H_Aarr_of_Rs
                                );

void build_sparse_transition_model_at_T_at_a(int t, int action_id, int bDimx, thrust::device_vector<float> &D_tdummy, 
                                float* D_all_u_arr, float* D_all_v_arr, float* D_all_ui_arr,
                                float*  D_all_vi_arr, float*  D_all_yi_arr,
                                float* D_all_s_arr, int* D_all_mask_arr,
                                thrust::device_vector<float> &D_params, 
                                thrust::host_vector<float> &H_params,
                                thrust::device_vector<float> &D_xs, 
                                thrust::device_vector<float> &D_ys, 
                                float** H_actions,
                                thrust::host_vector<int32_t> &H_coo_len_per_ac,
                                thrust::host_vector<long long int>* H_Aarr_of_cooS1,
                                thrust::host_vector<long long int>* H_Aarr_of_cooS2,
                                thrust::host_vector<float>* H_Aarr_of_cooProb,
                                thrust::host_vector<float>* H_Aarr_of_Rs
                                ){


    int gsize = (int) H_params[0];
    int num_actions =  (int)H_params[1];
    int nrzns = (int) H_params[2];
    int nt = (int) H_params[10];
    int m = (int)H_params[18];
    int Nb = (m*m) + 1; //+1 is to store no. of S2s not lying in nieghbour_array. Ideally it should have 0
                               
    // raw pointer casts
    float* D_T_arr = thrust::raw_pointer_cast(&D_tdummy[0]);
    float* xs = thrust::raw_pointer_cast(&D_xs[0]);
    float* ys = thrust::raw_pointer_cast(&D_ys[0]);
    float* params = thrust::raw_pointer_cast(&D_params[0]);

    //Define Kernel launch parameters for transition calculation kernel
    int DimGrid_z = (nrzns/bDimx)+1;
    if (nrzns % bDimx == 0)
        DimGrid_z = (nrzns/bDimx);
    
    // initialse master S2 array
    thrust::device_vector<float> D_master_S2_vector(ncells * Nb, 0);
    float* D_master_S2_arr = thrust::raw_pointer_cast(&D_master_S2_vector[0]);
    
    // initialise master sum_Rsa array - sumRsa's 
    // Important to initialise it with 0
    thrust::device_vector<float> D_master_sumRsa_vector(ncells, 0);
    float* D_master_sumRsa_arr = thrust::raw_pointer_cast(&D_master_sumRsa_vector[0]);

    // define kerel block and grid configuration
    dim3 DimGrid(gsize, gsize, DimGrid_z);
    dim3 DimBlock(bDimx, 1, 1);

    float ac_speed = H_actions[action_id][0];
    float ac_angle = H_actions[action_id][1];

    // launch kernel for @a @t
    transition_calc<<< DimGrid, DimBlock >>> (D_T_arr, 
        ncells, D_all_u_arr, D_all_v_arr, D_all_ui_arr, D_all_vi_arr, D_all_yi_arr,
        D_all_s_arr, D_all_mask_arr,
        ac_speed, ac_angle, xs, ys, params, D_master_sumRsa_arr, 
        D_master_S2_arr);

    cudaDeviceSynchronize();

    // checks
    // std::cout << "D_xs= " << std::endl;
    // for (int i = 0; i< 10; i++)
    //     std::cout << D_xs[i] << " " ;

    // if (t == nt-2){
    //     std::cout << "t = " << t << "\n nt = " << nt << "\n" ; 
    //     std::cout<<"gisze= " << gsize << std::endl;
    //     std::cout<<"DimGrid_z = " << DimGrid_z << std::endl;
    //     std::cout<<"bDimx = " <<  bDimx << std::endl;
    // }

    // // CHECK copy data back to host for check
    // std::cout << "a" << n <<"\n vx at s1=0: " << D_params[31] << std::endl;
    // std::cout <<"\n vx at s1=0: " << D_params[30] << std::endl;
    // std::cout << "----a" << n <<"\n";
    // std::cout <<"pre move " << "\n";
    // std::cout<<"r1\n"<< D_params[23] << "," << D_params[24] << std::endl;
    // std::cout<<"r2\n"<< D_params[25] << "," << D_params[26] << std::endl;
    // std::cout <<"post move " << "\n";
    // std::cout<<"r1\n"<< D_params[27] << "," << D_params[28] << std::endl;
    // std::cout<<"r2\n"<< D_params[29] << "," << D_params[30] << std::endl;

    // thrust::copy(D_master_S2_vector.begin() + n*arr_size, D_master_S2_vector.begin() + (n+1)*arr_size, H_S2_vec.begin());
    // thrust::copy(D_master_sumRsa_vector.begin() + n*ncells, D_master_sumRsa_vector.begin() + (n+1)*ncells, H_sumR_sa.begin());
    // std::cout << "post kernel" << std::endl;
    // for(int i = 0; i < 10; i ++)
    //     std::cout << H_sumR_sa[i] << std::endl;
    // for(int i = 0; i < 10; i ++)
    //     std::cout << H_S2_vec[i] << std::endl;
    

    int Nthreads = D_master_sumRsa_vector.size();
    assert(Nthreads == ncells);

    int threads_per_block = 64;
    int blocks_per_grid = (Nthreads/threads_per_block) + 1;
    assert( blocks_per_grid * threads_per_block >= Nthreads);
    
    compute_mean<<< blocks_per_grid, threads_per_block >>>(D_master_sumRsa_arr, Nthreads, nrzns);

    // TODO: in optimazation phase move this line after initilisation num_uq_S2 vectors.
    // cudaDeviceSynchronize();

    //initialising vectors for counting nnzs or number of uniqe S2s for S1s
    thrust::device_vector<unsigned long long int> D_num_uq_s2(ncells,0);
    thrust::device_vector<unsigned long long int> D_prSum_num_uq_s2(ncells);
    unsigned long long int* num_uq_s2_ptr = thrust::raw_pointer_cast(&D_num_uq_s2[0]);
    unsigned long long int* prSum_num_uq_s2_ptr = thrust::raw_pointer_cast(&D_prSum_num_uq_s2[0]);
    //one thread per element
    // count no. of ug unique S2 for each S1 and fill in num_uq_s2
    count_kernel<<<ncells, Nb>>>(D_master_S2_arr, nrzns, num_uq_s2_ptr);
    cudaDeviceSynchronize();

        //CHECKs
        // std::cout << "D_num_uq_s2_pc\n";
        // int tempflag = 0;
        // int tempnum;
        // int cnt2 = 0;
        // int cnt1 = 0;
        // for (int i =0; i < efCszNa; i++){
        //     tempnum = D_num_uq_s2_pc[i];
        //     if (tempnum == 1)
        //         cnt1++;
        //     else if (tempnum == 2)
        //         cnt2++;
        //     else
        //         std::cout << " --------------------------- WRONG-----------\n";
        // }
        // std::cout << "cnt1 = " << cnt1 << "\ncnt2 = " << cnt2 <<"\n";


    // calc nnz: number of non zero elements(or unique S2s) for a given S1 and action
    long long int nnz = thrust::reduce(D_num_uq_s2.begin(), D_num_uq_s2.end(), (float) 0, thrust::plus<float>());
    // get prefix sum of D_num_uq_s2. This helps threads to access apt COO indices in reduce_kernel
    thrust::exclusive_scan(D_num_uq_s2.begin(), D_num_uq_s2.end(), D_prSum_num_uq_s2.begin());
    // std::cout << "nnz = " << nnz<< "\n";

    //initilise coo arrays (concated across actions)
    thrust::device_vector<long long int> D_coo_s1(nnz);
    thrust::device_vector<long long int> D_coo_s2(nnz);
    thrust::device_vector<float> D_coo_count(nnz); // TODO: makde this int32_t and introduce another array for prob
    long long int* D_coo_s1_arr = thrust::raw_pointer_cast(&D_coo_s1[0]);
    long long int* D_coo_s2_arr = thrust::raw_pointer_cast(&D_coo_s2[0]);
    float* D_coo_cnt_arr = thrust::raw_pointer_cast(&D_coo_count[0]);

    Nthreads = ncells;
    assert(Nthreads == ncells);
    threads_per_block = 64;
    blocks_per_grid = (Nthreads/threads_per_block) + 1;
    // reduce operation to fill COO arrays
    reduce_kernel<<<blocks_per_grid, threads_per_block>>>(D_master_S2_arr, t, Nb, m,
                                ncells, nrzns, gsize, D_coo_s1_arr, D_coo_s2_arr, D_coo_cnt_arr, 
                                num_uq_s2_ptr, prSum_num_uq_s2_ptr);
    cudaDeviceSynchronize();


    // nnz should be filled in a global array 
    H_coo_len_per_ac[action_id] = nnz;
    // Copy Device COO rusults to Host COO vectors across actions and append vectors across time
    assert(action_id >=0);
    H_Aarr_of_cooS1[action_id].insert(H_Aarr_of_cooS1[action_id].end(), D_coo_s1.begin(), D_coo_s1.end());
    H_Aarr_of_cooS2[action_id].insert(H_Aarr_of_cooS2[action_id].end(), D_coo_s2.begin(), D_coo_s2.end());
    H_Aarr_of_cooProb[action_id].insert(H_Aarr_of_cooProb[action_id].end(), D_coo_count.begin(), D_coo_count.end());
    H_Aarr_of_Rs[action_id].insert(H_Aarr_of_Rs[action_id].end(), D_master_sumRsa_vector.begin(), D_master_sumRsa_vector.end());

        //checks

        // print_device_vector(D_coo_s1, 0, 10, "D_coo_s1", " ", 0);
        // print_device_vector(D_coo_s2, 0, 10, "D_coo_s2", " ", 0);

        // std::cout << "H_coo_len_per_ac" << std::endl;
        // for (int n = 0; n < num_actions; n++)
        //   std::cout << H_coo_len_per_ac[n] << std::endl;

        // std::cout << "H_Aarr_of_cooS1" << std::endl;
        // for (int n = 0; n < num_actions; n++){
        //     for (int i = 0; i < H_Aarr_of_cooS1[n].size(); i++)
        //         std::cout << H_Aarr_of_cooS1[n][i] << " , " << H_Aarr_of_cooS2[n][i] << " , " << H_Aarr_of_cooProb[n][i] << std::endl;
        //     std::cout << std::endl;
        // }

        // std::cout << "H_Aarr_of_Rs" << std::endl;
        // for (int n = 0; n < num_actions; n++){
        //     for (int i = 0; i < ncells; i++)
        //         std::cout << H_Aarr_of_Rs[n][i] << std::endl;
        //     std::cout << std::endl;
        // }

        // // array of num_actions decive_vvectors for sum_Rsa_vec
        // // initialasation with 0 is important. because values are added to this
        // thrust::host_vector<float> H_arr_sumR_sa[num_actions];
        // for(int n = 0; n < num_actions; n++){
        //     H_arr_sumR_sa[n] = thrust::host_vector<float>(nnz[i]);
    

}





int get_reward_type(std::string prob_type){
    // returns 
    // 0 for time
    // 1 for energy1
    // 2 for energy2
    // 3 for energy3

    if (prob_type == "time")
        return 0;
    else if (prob_type == "energy1")
        return 1;
    else if (prob_type == "energy2")
        return 2;
    else if (prob_type == "energy3")
        return 3;
    else    
        return -1;
}


// ------------------------------- MAIN ()  ---------------------------------




int main(){

    // -------------------- input data starts here ---------------------------------


    #include "input_to_build_model.h"
 
    int reward_type = get_reward_type(prob_type);
    std::cout << "Reward type: " << reward_type << "\n";

    // define full problem name and print them to a temporary file
    // the temp file will be read by python scripts for conversion
    std::string prob_specs = get_prob_name(num_ac_speeds, num_ac_angles, i_term, 
                                            j_term, term_subgrid_size);
    std::string op_Fname_upto_prob_name = "data_modelOutput/" + prob_type + "/"
                                 + prob_name + "/" ;
    std::string op_FnamePfx = op_Fname_upto_prob_name + prob_specs + "/"; //path for storing op npy data.
    std::ofstream fout("temp_modelOp_dirName.txt");
    fout << prob_type << "\n";
    fout << prob_name << "\n";
    fout << prob_specs << "\n";
    fout << op_FnamePfx;
    fout.close();

    // TODO:  Make sure files are stored in np.float32 format
    std::string data_path = "data_input/" + prob_name + "/";
    std::string all_u_fname = data_path + "all_u_mat.npy";
    std::string all_v_fname = data_path + "all_v_mat.npy";
    std::string all_ui_fname = data_path + "all_ui_mat.npy";
    std::string all_vi_fname = data_path + "all_vi_mat.npy";
    std::string all_yi_fname = data_path + "all_Yi.npy";
    std::string all_s_fname = data_path + "all_s_mat.npy";
    std::string all_mask_fname = data_path + "obstacle_mask.npy"; //this file stored in int32


    // -------------------- input data ends here ---------------------------------


    // make directory for storing output data from this file
    make_dir(op_Fname_upto_prob_name);
    make_dir(op_FnamePfx);

    int all_u_n_elms;
    int all_v_n_elms;
    int all_ui_n_elms;
    int all_vi_n_elms;
    int all_yi_n_elms;
    int all_s_n_elms;
    int all_mask_n_elms;

    cnpy::NpyArray all_u_cnpy = read_velocity_field_data(all_u_fname, &all_u_n_elms);
    cnpy::NpyArray all_v_cnpy = read_velocity_field_data(all_v_fname, &all_v_n_elms);
    cnpy::NpyArray all_ui_cnpy = read_velocity_field_data(all_ui_fname, &all_ui_n_elms);
    cnpy::NpyArray all_vi_cnpy = read_velocity_field_data(all_vi_fname, &all_vi_n_elms);
    cnpy::NpyArray all_yi_cnpy = read_velocity_field_data(all_yi_fname, &all_yi_n_elms);
    cnpy::NpyArray all_s_cnpy = read_velocity_field_data(all_s_fname, &all_s_n_elms);
    cnpy::NpyArray all_mask_cnpy = read_velocity_field_data(all_mask_fname, &all_mask_n_elms);


    float* all_u_mat = all_u_cnpy.data<float>();
    float* all_v_mat = all_v_cnpy.data<float>();
    float* all_ui_mat = all_ui_cnpy.data<float>();
    float* all_vi_mat = all_vi_cnpy.data<float>();
    float* all_yi_mat = all_yi_cnpy.data<float>();
    float* all_s_mat = all_s_cnpy.data<float>();
    int* all_mask_mat = all_mask_cnpy.data<int>();

    // CHECKS:
    // print_array<float>(all_u_mat, all_u_n_elms, "all_u_mat", " ");
    // print_array<float>(all_ui_mat, all_ui_n_elms,"all_ui_mat", " ");
    // print_array<float>(all_yi_mat, all_yi_n_elms,"all_yi_mat", " ");

    std::cout << "Finished reading Velocity Field Data !" << std::endl;
    assert(neighb_gsize <= gsize);
    assert((int)neighb_gsize%2 == 1);    // neighb_gsize must be odd

    //TODO: fill params in a function
    // Contains implicit casting from int32_t to float
    thrust::host_vector<float> H_params(32);
    H_params[0] = gsize;
    H_params[1] = num_actions; 
    H_params[2] = nrzns;
    H_params[3] = F;
    H_params[4] = dt;
    H_params[5] = r_outbound;
    H_params[6] = r_terminal;
    H_params[7] = nmodes;
    H_params[8] = i_term;
    H_params[9] = j_term;
    H_params[10] = nt;
    H_params[11] = is_stationary;
    H_params[12] = term_subgrid_size;
    H_params[13] = reward_type;
    H_params[14] = num_ac_speeds;
    H_params[15] = num_ac_angles;
    H_params[16] = dx;
    H_params[17] = dy;
    H_params[18] = neighb_gsize; // referred to as m in functions

    for( int i =20; i<32; i++)
        H_params[i] = z;

    // Define grid ticks in host
    thrust::host_vector<float> H_xs(gsize, -1);
    thrust::host_vector<float> H_ys(gsize, -1);
    float* xs = thrust::raw_pointer_cast(&H_xs[0]);
    float* ys = thrust::raw_pointer_cast(&H_ys[0]);

    define_xs_or_ys(xs, dx, x0, gsize);
    define_xs_or_ys(ys, dy, y0, gsize);

    // define angles in host
    float** H_actions = new float*[num_actions];
    for(int i=0; i<num_actions; i++)
        H_actions[i] = new float[2];
    populate_actions(H_actions, num_ac_speeds, num_ac_angles, F);
    // std::cout << "CHECK:   ACTIONS:    \n";
    // for(int i=0; i<num_actions; i++){
    //     std::cout << H_actions[i][0] << ", " << H_actions[i][1] << "\n";
    // }



    //----- start copying data to device --------

    // Copy vel field data to device memory using thrust device_vector
    thrust::device_vector<float> D_all_u_vec (all_u_mat, all_u_mat + all_u_n_elms);
    thrust::device_vector<float> D_all_v_vec (all_v_mat, all_v_mat + all_v_n_elms);
    thrust::device_vector<float> D_all_ui_vec (all_ui_mat, all_ui_mat + all_ui_n_elms);
    thrust::device_vector<float> D_all_vi_vec (all_vi_mat, all_vi_mat + all_vi_n_elms);
    thrust::device_vector<float> D_all_yi_vec (all_yi_mat, all_yi_mat + all_yi_n_elms);
    thrust::device_vector<float> D_all_s_vec (all_s_mat, all_s_mat + all_s_n_elms);
    thrust::device_vector<int> D_all_mask_vec (all_mask_mat, all_mask_mat + all_mask_n_elms);


    float* D_all_u_arr = thrust::raw_pointer_cast(&D_all_u_vec[0]);
    float* D_all_v_arr = thrust::raw_pointer_cast(&D_all_v_vec[0]);
    float* D_all_ui_arr = thrust::raw_pointer_cast(&D_all_ui_vec[0]);
    float* D_all_vi_arr = thrust::raw_pointer_cast(&D_all_vi_vec[0]);
    float* D_all_yi_arr = thrust::raw_pointer_cast(&D_all_yi_vec[0]);
    float* D_all_s_arr = thrust::raw_pointer_cast(&D_all_s_vec[0]);
    int* D_all_mask_arr = thrust::raw_pointer_cast(&D_all_mask_vec[0]);


    std::cout << "Copied to Device : Velocity Field Data !" << std::endl;

    thrust::device_vector<float> D_tdummy(2,0);
    // initialise empty device vectors. These contain time-invariant data
    thrust::device_vector<float> D_params(32);
    thrust::device_vector<float> D_xs(gsize);
    thrust::device_vector<float> D_ys(gsize);

    // initialise reuseable host vectors
    thrust::host_vector<int32_t> H_coo_len_per_ac(num_actions);
    thrust::host_vector<long long int> H_Aarr_of_cooS1[(int)num_actions];
    thrust::host_vector<long long int> H_Aarr_of_cooS2[(int)num_actions];
    thrust::host_vector<float> H_Aarr_of_cooProb[(int)num_actions];
    thrust::host_vector<float> H_Aarr_of_Rs[(int)num_actions];

    //initialised with 0 size. later data from device is inserted/appended to the end of vector
    for (int i =0; i < num_actions; i++){
        H_Aarr_of_cooS1[i] = thrust::host_vector<long long int> (0);
    }
    for (int i =0; i < num_actions; i++){
        H_Aarr_of_cooS2[i] = thrust::host_vector<long long int> (0);
    }
    for (int i =0; i < num_actions; i++){
        H_Aarr_of_cooProb[i] = thrust::host_vector<float> (0);
    }
    for (int i =0; i < num_actions; i++){
        H_Aarr_of_Rs[i] = thrust::host_vector<float> (0);
    }

    // assign value to global variable
    ncells = gsize*gsize;           

    // copy data from host to device
    D_params = H_params;
    D_xs = H_xs;
    D_ys = H_ys;

    // run time loop and compute transition data for each time step
    auto start = high_resolution_clock::now(); 
    auto end = high_resolution_clock::now(); 
    auto duration_t = duration_cast<microseconds>(end - start);

    //IMP: Run time loop till nt-1. There ar no S2s to S1s in the last timestep
    for(int t = 0; t < nt-1; t++){
        std::cout << "*** Computing data for timestep, T = " << t << std::endl;
        D_tdummy[0] = t;
        start = high_resolution_clock::now(); 
            for(int action_id = 0; action_id < num_actions; action_id++){
                // std::cout << "  * action_id= " << action_id;
                
                // this function also concats coos across time.
                build_sparse_transition_model_at_T_at_a(t, action_id, bDimx, D_tdummy, D_all_u_arr, D_all_v_arr, 
                        D_all_ui_arr, D_all_vi_arr, D_all_yi_arr,
                        D_all_s_arr, D_all_mask_arr,
                        D_params, H_params, D_xs, D_ys, H_actions, 
                        H_coo_len_per_ac,
                        H_Aarr_of_cooS1, H_Aarr_of_cooS2, H_Aarr_of_cooProb,
                        H_Aarr_of_Rs);
                        //  output_data )  
            }
        end = high_resolution_clock::now(); 
        std::cout << std::endl ;
        duration_t = duration_cast<microseconds>(end - start);
        std::cout << "duration@t = "<< duration_t.count()/1e6 << "sec" << std::endl;
        std::cout << 3*H_Aarr_of_cooS1[0].size()*4*1e-6 << " MB" << std::endl;
        std::cout << std::endl << std::endl;
    }


    // fill R vectors of each action for the last time step with high negative values. 
    // this has to be done seaprately because the above loop runs till nt-1.
    /*
        TODO: 1. Verify rewards as last time step
    */
    thrust::host_vector<float> H_rewards_at_end_t(ncells, 0);
    for (int i =0; i < num_actions; i++){
        H_Aarr_of_Rs[i].insert(H_Aarr_of_Rs[i].end(), H_rewards_at_end_t.begin(), H_rewards_at_end_t.end());
    }

    //Check
    // for (int i =0; i < num_actions; i++)
    //     std::cout << H_Aarr_of_Rs[i].size() << " ";
    

    // find nnz per action
    thrust::host_vector<long long int> H_master_PrSum_nnz_per_ac(num_actions);
    long long int DP_relv_params[2] = {ncells*nt, num_actions*1LL};

    long long int master_nnz = 0;       //running sum of nnz going across actions

    // calculate inclusive prefix sum of nnz's across actions 
    // will be used to access indeces while concatenating results across across actions
    for(int i = 0; i < num_actions; i++){
        master_nnz += H_Aarr_of_cooS1[i].size();
        H_master_PrSum_nnz_per_ac[i] = master_nnz;
    }

    unsigned long int num_DP_params = sizeof(DP_relv_params) / sizeof(DP_relv_params[0]);
    // print_array<long long int>(DP_relv_params, 2, "DP_relv_params", " ");
    // std::cout << "chek num = " << sizeof(DP_relv_params) << std::endl;
    // std::cout << "chek denom = " << sizeof(DP_relv_params[0]) << std::endl;

    //checks
    // std::cout << "total/master_nnz = " << master_nnz << std::endl;
    // std::cout << "H_Aarr_of_cooS1[i].size()" << std::endl;
    // for(int i = 0; i < num_actions; i++)
    //     std::cout << H_Aarr_of_cooS1[i].size() << std::endl;
    // print_array<long long int>(&H_Aarr_of_cooS2[0][0], 10,  "H_Aarr_of_cooS2[0]", " ");


    // save final coo data
    thrust::host_vector<long long int> H_master_cooS1(master_nnz);
    thrust::host_vector<long long int> H_master_cooS2(master_nnz);
    thrust::host_vector<float> H_master_cooVal(master_nnz);
    thrust::host_vector<float> H_master_R(ncells*nt*num_actions, -99999); //TODO: veriffy -99999
    save_master_Coos_to_file(op_FnamePfx, num_actions,
                                H_master_cooS1, 
                                H_master_cooS2, 
                                H_master_cooVal,
                                H_master_R,
                                H_Aarr_of_cooS1,
                                H_Aarr_of_cooS2,
                                H_Aarr_of_cooProb,
                                H_Aarr_of_Rs,
                                H_params,
                                DP_relv_params,
                                num_DP_params);


    return 0;
}

//------------ main ends here ------------------------------------------





void save_master_Coos_to_file(std::string op_FnamePfx, int num_actions,
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
    unsigned long int num_DP_params){
    //  Convertes floats to int32 for COO row and col idxs
    //  copies from each action vector to a master vector
    //  master_coo vectors is concatation first across time, then across action
    //  ALSO, MODIFIES S1(t,i,j) to S1(t,i,j,a)

    unsigned long long int master_nnz = H_master_cooS1.size();
    unsigned long long int prob_params_size = prob_params.size();
    long long int m_idx = 0;
    int n_states = DP_relv_params[0];


    for(int i = 0; i < num_actions; i++){
        for(int j = 0; j< H_Aarr_of_cooS1[i].size(); j++){
            // TODO: modify to include actions
            H_master_cooS1[m_idx] = H_Aarr_of_cooS1[i][j] + i*n_states;
            m_idx++;
        }
    }

    m_idx = 0;
    for(int i = 0; i < num_actions; i++){
        for(int j = 0; j< H_Aarr_of_cooS2[i].size(); j++){
            H_master_cooS2[m_idx] = H_Aarr_of_cooS2[i][j];
            m_idx++;
        }
    }

    m_idx = 0;
    for(int i = 0; i < num_actions; i++){
        for(int j = 0; j< H_Aarr_of_cooProb[i].size(); j++){
            H_master_cooVal[m_idx] = H_Aarr_of_cooProb[i][j];
            m_idx++;
        }
    }

    m_idx = 0;
    for(int i = 0; i < num_actions; i++){
        for(int j = 0; j< H_Aarr_of_Rs[i].size(); j++){
            H_master_R[m_idx] = H_Aarr_of_Rs[i][j];
            m_idx++;
        }
    }

    
    std::cout << "check num_DP_params = " << num_DP_params << std::endl;
    std::cout << "op_FnamePfx= " <<  op_FnamePfx << "\n";
    
    cnpy::npy_save(op_FnamePfx + "master_cooS1.npy", &H_master_cooS1[0], {master_nnz,1},"w");
    cnpy::npy_save(op_FnamePfx + "master_cooS2.npy", &H_master_cooS2[0], {master_nnz,1},"w");
    cnpy::npy_save(op_FnamePfx + "master_cooVal.npy", &H_master_cooVal[0], {master_nnz,1},"w");
    cnpy::npy_save(op_FnamePfx + "master_R.npy", &H_master_R[0], {H_master_R.size(),1},"w");
    cnpy::npy_save(op_FnamePfx + "DP_relv_params.npy", &DP_relv_params[0], {num_DP_params,1},"w");
    cnpy::npy_save(op_FnamePfx + "prob_params.npy", &prob_params[0], {prob_params_size,1},"w");

}



cnpy::NpyArray read_velocity_field_data( std::string file_path_name, int* n_elements){
    // reads numpy file from input and 
    // returns cnpy::NpyArray stucture  and also fills in num_elements in the passed reference n_elements
    // extraction in main: float* vel_data = arr.data<float>();
    // TODO: make it general. currently hard-coded for float arrays.

    //print filename
    std::cout << "file path and name:   " << file_path_name << std::endl;
    cnpy::NpyArray arr = cnpy::npy_load(file_path_name);

    //prints for checks 
    int dim = arr.shape.size();
    int num_elements = 1;
    std::cout << "shape: " ;
    for (int i = 0; i < dim; i++){
        std::cout << arr.shape[i] << " , " ;
        num_elements = num_elements*arr.shape[i];
    }
    *n_elements = num_elements;
    std::cout << std::endl << "num_elements: " << num_elements << std::endl;

    float* vel_data = arr.data<float>();
    // print check first 10 elements
    std::cout << "First 10 elements of loaded array are: " << std::endl;
    for (int i = 0; i < 10; i++)
         std::cout << vel_data[i] << "  " ;
    
    std::cout << std::endl << std::endl;

    return arr;

}

/*
--- print_device_vector()  ---
IMP: datatype has to be explicityle changed in that file
*/

// template<typename dType>
void print_device_vector( thrust::device_vector<long long int> &array, int start_id, int end_id, std::string array_name, std::string end, int method){
    std::cout << array_name << "  from id " << start_id << "  to  " << end_id << std::endl;
    if (method == 1){
        float temp = -10000000;
        for(int i = start_id; i < end_id; i++){
            if (array[i] != temp){
                std::cout << i << "\n";
                std::cout << array[i] << " " << end;
                std::cout << "\n";
                temp = array[i];
            }
        }
    }

    else if (method == 0){
        for(int i = start_id; i < end_id; i++)
            std::cout << array[i] << " " << end;
    }

    else
        std::cout << "Invalid input for argument: method";


    std::cout << std::endl;
}


void make_dir(std::string dir_name){
    int mkdir_status;
    std::string comm_mkdir = "mkdir ";
    std::string str = comm_mkdir + dir_name;
    const char * full_command = str.c_str();
    mkdir_status = system(full_command);
    std::cout << "mkdir_status = " << mkdir_status << std::endl;
}



void define_xs_or_ys(float* xs, float dx, float x0, int gsize){

    for(int i = 0; i < gsize;  i++)
        xs[i] = x0 + i*dx;
}



void populate_ac_angles(float* ac_angles, int num_ac_angles){
    //fills array with equally spaced angles in radians
    for (int i = 0; i < num_ac_angles; i++)
        ac_angles[i] = i*(2*M_PI)/num_ac_angles;
    return;
}



void populate_ac_speeds(float* ac_speeds, int num_ac_speeds, float Fmax){
    //fills array with ac_speeds
    // std::cout << "infunc CHeck- num_ac_speeds = " << num_ac_speeds << "\n";
    float delF = 0;
    if (num_ac_speeds == 1)
        ac_speeds[0] = Fmax;
    else if (num_ac_speeds > 1){
        // -----include 0 speed
        // delF = Fmax/(num_ac_speeds-1);
        // for(int i = 0; i<num_ac_speeds; i++)
        //     ac_speeds[i] = i*delF;
        // ------exclude 0 speed
        delF = Fmax/(num_ac_speeds);
        for(int i = 0; i<num_ac_speeds; i++){
            ac_speeds[i] = (i+1)*delF;
            // std::cout << ac_speeds[i] << "\n";
        }
    }
    else
        std::cout << "Invalid num_ac_speeds\n";
    
    return;
}



void populate_actions(float **H_actions, int num_ac_speeds, int num_ac_angles, float Fmax){
    // populates 2d vector with possible actions
    float* ac_angles = new float[num_ac_angles];
    populate_ac_angles(ac_angles, num_ac_angles);

    float* ac_speeds = new float[num_ac_speeds];
    populate_ac_speeds(ac_speeds, num_ac_speeds, Fmax);

    int idx;
    for (int i=0; i<num_ac_speeds; i++){
        for(int j=0; j<num_ac_angles; j++){
            idx = j + num_ac_angles*i;
            // std::cout << ac_speeds[i] << "\n";
            H_actions[idx][0] = ac_speeds[i];
            H_actions[idx][1] = ac_angles[j];
        }
    }

    return;
}