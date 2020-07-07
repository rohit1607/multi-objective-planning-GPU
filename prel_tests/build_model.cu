#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/unique.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include "cnpy.h"
// #include "read_npy_files.h"
// #include "grid.h"
#include <cmath>


#include <iostream>



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

__device__ int32_t state1D_from_thread(int32_t T){   
    // j ~ blockIdx.x
    // i ~ blockIdx.y 
    // The above three consitute a spatial state index from i and j of grid
    // last term is for including time index as well.
    return (blockIdx.x + (blockIdx.y*gridDim.x) + (T*gridDim.x*gridDim.y) ); 
}


__device__ int32_t state1D_from_ij(int32_t*  posid, int32_t T){
    // posid = {i , j}
    // state id = j + i*dim(i) + T*dim(i)*dim(j)
    return (posid[1] + posid[0]*gridDim.x + (T*gridDim.x*gridDim.y) ) ; 
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
    int32_t i_term = params[8];         // terminal state indices
    int32_t j_term = params[9];
    if(i == i_term && j == j_term)
        return true;
    else return false;
}


__device__ bool my_isnan(int s){
    // By IEEE 754 rule, NaN is not equal to NaN
    return s != s;
}


__device__ void get_xypos_from_ij(int32_t i, int32_t j, float* xs, float* ys, float* x, float* y){
    *x = xs[j];
    *y = ys[gridDim.x - 1 - i];
    return;
}


__device__ float get_angle_in_0_2pi(float theta){
    float f_pi = 3.141592;
    if (theta < 0)
        return theta + (2*f_pi);
    else
        return theta;

}


__device__ float calculate_reward_const_dt(float* xs, float* ys, int32_t i_old, int32_t j_old, float xold, float yold, int32_t* newposids, float* params, float vnet_x, float vnet_y ){
    // xold and yold are centre of old state (i_old, j_old)
    float dt = params[4];
    float r1, r2, theta1, theta2, theta, h;
    float dt_new;
    float xnew, ynew;
    if (newposids[0] == i_old && newposids[1] == j_old)
        dt_new = dt;
    else
    {
        get_xypos_from_ij(newposids[0], newposids[1], xs, ys, &xnew, &ynew); //get centre of new states
        h = sqrtf((xnew - xold)*(xnew - xold) + (ynew - yold)*(ynew - yold));
        r1 = h/(sqrtf((vnet_x*vnet_x) + (vnet_y*vnet_y)));
        theta1 = get_angle_in_0_2pi(atan2f(vnet_y, vnet_x));
        theta2 = get_angle_in_0_2pi(atan2f(ynew - yold, xnew - xold));
        theta = fabsf(theta1 -theta2);
        r2 = fabsf(sinf(theta));
        dt_new = r1 + r2;
        if (threadIdx.x == 0 && blockIdx.z == 0 && blockIdx.x == 1 && blockIdx.y == 1)
        {
            params[24] = r1;
            params[25] = r2;
        }
    }
    return -dt_new;
}


__device__ void move(float ac_angle, float vx, float vy, float* xs, float* ys, int32_t* posids, float* params, float* r ){
    int32_t n = params[0] - 1;      // gsize - 1
    // int32_t num_actions = params[1];
    // int32_t nrzns = params[2];
    float F = params[3];
    float dt = params[4];
    float r_outbound = params[5];
    float r_terminal = params[6];
    float Dj = fabsf(xs[1] - xs[0]);
    float Di = fabsf(ys[1] - ys[0]);
    float r_step = 0;
    *r = 0;
    int32_t i0 = posids[0];
    int32_t j0 = posids[1];
    float vnetx = F*cosf(ac_angle) + vx;
    float vnety = F*sinf(ac_angle) + vy;
    float x, y;
    get_xypos_from_ij(i0, j0, xs, ys, &x, &y); // x, y stores centre coords of state i0,j0
    float xnew = x + (vnetx * dt);
    float ynew = y + (vnety * dt);
    //checks TODO: remove checks once verified
    if (threadIdx.x == 0 && blockIdx.z == 0 && blockIdx.x == 1 && blockIdx.y == 1)
    {
        params[12] = x;
        params[13] = y;
        params[14] = vnetx;
        params[15] = vnety;
        params[16] = xnew;
        params[17] = ynew;
        params[18] = ac_angle;
    }
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
    // TODO:xxDONE check logic wrt remainderf. remquof had issue
    int32_t xind, yind;
    //float remx = remquof((xnew - xs[0]), Dj, &xind);
    //float remy = remquof(-(ynew - ys[n]), Di, &yind);
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
            posids[0] = yind;
            posids[1] = xind;
            if (is_edge_state(posids[0], posids[1]))     //line 110
                {
                    *r += r_outbound;
                }
            
            if (threadIdx.x == 0 && blockIdx.z == 0 && blockIdx.x == 1 && blockIdx.y == 1)
            {
                params[26] = 9999;
            }
        }
    // r_step = calculate_reward_const_dt(xs, ys, i0, j0, x, y, posids, params, vnetx, vnety);
    //TODO: change back to normal when needed
    r_step = -dt;
    *r += r_step; //TODO: numerical check remaining
    if (is_terminal(posids[0], posids[1], params))
        {
            *r += r_terminal;
        }
    
    if (threadIdx.x == 0 && blockIdx.z == 0 && blockIdx.x == 1 && blockIdx.y == 1)
    {
        params[19] = xnew;
        params[20] = ynew;
        params[21] = yind;
        params[22] = xind;
        params[23] = *r;
        //params[17] = ynew;
        //params[18] = ac_angle;
    }
}


__device__ void extract_velocity(float* vx, float* vy, int32_t T, float* all_u_mat, float* all_v_mat, float* all_ui_mat, float* all_vi_mat, float* all_Yi, float* params){
    int32_t nrzns = params[2];
    int32_t nmodes = params[7];              

    int32_t sp_uvi, str_uvi, sp_Yi, str_Yi; //startpoints and strides for accessing all_ui_mat, all_vi_mat and all_Yi
    float sum_x = 0;
    float sum_y = 0;
    float vx_mean, vy_mean;
    //thread index. also used to access resultant vxrzns[nrzns, gsize, gsize]
    int32_t idx = get_thread_idx();
    //rzn index to identify which of the 5k rzn it is. used to access all_Yi.
    int32_t rzn_id = (blockIdx.z * blockDim.x)  + threadIdx.x ;
    //mean_id is the index used to access the flattened all_u_mat[t,i,j].
    int32_t mean_id = state1D_from_thread(T);
    //to access all_ui_mat and all_vi_mat
    str_uvi = gridDim.x * gridDim.y;
    sp_uvi = (T * nmodes * str_uvi) + (gridDim.x * blockIdx.y) + (blockIdx.x);
    // to access all_Yi
    sp_Yi = (T * nrzns * nmodes) + (rzn_id * nmodes);
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


//test: changer from float* to float ac_angle
__global__ void transition_calc(float* T_arr, float* all_u_mat, float* all_v_mat, float* all_ui_mat, float* all_vi_mat, float* all_Yi,
                            float ac_angle, float* xs, float* ys, float* params, float* sumR_sa, float* results)
                                    // resutls directions- 1: along S2;  2: along S1;    3: along columns towards count
{
    int32_t gsize = params[0];          // size of grid along 1 direction. ASSUMING square grid.
    int32_t num_actions = params[1];    
    int32_t nrzns = params[2];
    float F = params[3];
    float dt = params[4];
    float r_outbound = params[5];
    float r_terminal = params[6];
    int32_t nmodes = params[7];              
    int32_t i_term = params[8];         // terminal state indices
    int32_t j_term = params[9];
    int32_t nT = params[10];
    int32_t is_stationary = params[11];
    int32_t T = (int32_t)T_arr[0];
    int32_t idx = get_thread_idx();
    float vx, vy;
    if(idx < gridDim.x*gridDim.y*nrzns)
    {
        int32_t posids[2] = {blockIdx.y, blockIdx.x};    //static declaration of array of size 2 to hold i and j values of S1. 
        int32_t sp_id;      //sp_id is space_id. S1%(gsize*gsize)
        //  Afer move() these will be overwritten by i and j values of S2
        float r;              // to store immediate reward
        extract_velocity(&vx, &vy, T, all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, all_Yi, params);
        //move(*ac_angle, vx, vy, xs, ys, posids, params, &r);
        move(ac_angle, vx, vy, xs, ys, posids, params, &r);
        int32_t S1, S2;
        if (is_stationary == 1)
        {
            T = 0;
            S1 = state1D_from_thread(T);     //get init state number corresponding to thread id
            S2 = state1D_from_ij(posids, T);   //get successor state number corresponding to posid and next timestep T+1        
        }
        else
        {
            S1 = state1D_from_thread(T);     //get init state number corresponding to thread id
            S2 = state1D_from_ij(posids, T+1);   //get successor state number corresponding to posid and next timestep T+1        
            sp_id = S1%(gsize*gsize);
        }
        //writing to sumR_sa. this array will later be divided by num_rzns, to get the avg
        float a = atomicAdd(&sumR_sa[sp_id], r); //TODO: try reduction if this is slow overall
        results[idx] = S2;
        __syncthreads();
        /*if (threadIdx.x == 0 && blockIdx.z == 0)
        {
            sumR_sa[S1] = sumR_sa[S1]/nrzns;    //TODO: change name to R_sa from sumR_sa since were not storing sum anymore
        }
    */
    }//if ends
    return;
}




void build_sparse_transition_model_at_T(int t, int bDimx, thrust::device_vector<float> &D_tdummy, 
                                        float* D_all_u_arr, float* D_all_v_arr, float* D_all_ui_arr,
                                        float*  D_all_vi_arr, float*  D_all_yi_arr,
                                        thrust::device_vector<float> &D_params, thrust::device_vector<float> &D_xs, 
                                        thrust::device_vector<float> &D_ys, thrust::device_vector<float> &D_ac_angles 
                                        //TODO: ,output_data 
                                        );
// void concatenate_results_across_time();


void build_sparse_transition_model_at_T(int t, int bDimx, thrust::device_vector<float> &D_tdummy, 
                                        float* D_all_u_arr, float* D_all_v_arr, float* D_all_ui_arr,
                                        float*  D_all_vi_arr, float*  D_all_yi_arr,
                                        thrust::device_vector<float> &D_params, thrust::device_vector<float> &D_xs, 
                                        thrust::device_vector<float> &D_ys, thrust::device_vector<float> &D_ac_angles 
                                        //TODO: ,output_data 
                                        ){

    int gsize = (int) D_params[0];
    int num_actions =  (int)D_params[1];
    int num_rzns = (int) D_params[2];

    // check velocity data and vector data
    std::cout << "D_paramas" << std::endl;
    for (int i = 0; i< 10; i ++)
        std::cout << D_params[i] << std::endl;

    // I think doing it this way does not issue a memcpy at the backend. thats why it fails
    // std::cout << "D_all_u_arr" << std::endl;
    // for (int i = 0; i< 10; i ++)
    //     std::cout << D_all_u_arr[i] << std::endl;                                 

    float* D_T_arr = thrust::raw_pointer_cast(&D_tdummy[0]);
    float* xs = thrust::raw_pointer_cast(&D_xs[0]);
    float* ys = thrust::raw_pointer_cast(&D_ys[0]);
    float* params = thrust::raw_pointer_cast(&D_params[0]);


    int ncells = gsize*gsize;
    thrust::host_vector<float> H_S2_vec(ncells*num_rzns, -1); //eqv of results
    thrust::host_vector<float> H_sumR_sa(ncells, 0);
    
    // array of num_actions device_vectors for S2_vec's
    thrust::device_vector<float> D_arr_S2vecs[num_actions];
    for(int n = 0; n < num_actions; n++){
        D_arr_S2vecs[n] = thrust::device_vector<float>(ncells*num_rzns, -1);
    }

    // array of num_actions decive_vvectors for sum_Rsa_vec
    // initialasation with 0 is important. because values are added to this
    thrust::device_vector<float> D_arr_sumR_sa[num_actions];
    for(int n = 0; n < num_actions; n++){
        D_arr_sumR_sa[n] = thrust::device_vector<float>(ncells, 0);
    }

    std::cout<<"gisze= " << gsize << std::endl;
    std::cout<<"g.z = " << (num_rzns/bDimx) + 1 << std::endl;
    std::cout<<"bDimx = " <<  bDimx << std::endl;
    dim3 DimGrid(gsize, gsize, (num_rzns/bDimx) + 1);
    dim3 DimBlock(bDimx, 1, 1);

    //launch kernel (n, t, rest_of_data) for each actions
    for(int n = 0; n < num_actions; n++){

        std::cout <<  std::endl <<"     a = " << n << std::endl;
        // TODO: ac_angles host vector instead of device
        float ac_angle = D_ac_angles[n];
        float* sumR_sa = thrust::raw_pointer_cast(&D_arr_sumR_sa[n][0]);
        float* S2_arr = thrust::raw_pointer_cast(&D_arr_S2vecs[n][0]);

        transition_calc<<< DimGrid, DimBlock  >>> (D_T_arr, D_all_u_arr, D_all_v_arr, D_all_ui_arr, D_all_vi_arr, D_all_yi_arr,
            ac_angle, xs, ys, params, sumR_sa, S2_arr);

        // cudaDeviceSynchronize();

        // copy data back to host
        H_S2_vec = D_arr_S2vecs[n];
        H_sumR_sa = D_arr_sumR_sa[n];
        std::cout << "post kernel" << std::endl;
        for(int i = 0; i < 10; i ++)
            std::cout << H_sumR_sa[i] << std::endl;
        for(int i = 0; i < 10; i ++)
            std::cout << H_S2_vec[i] << std::endl;
                    
    }

    cudaDeviceSynchronize();

    // Post processing

    // assignment to output data


}


cnpy::NpyArray read_velocity_field_data( std::string file_path_name, int* n_elements);

void define_xs_or_ys(float* xs, float dx, float x0, int gsize);

void populate_ac_angles(float* ac_angles, int num_actions);


int main(){

// -------------------- input data starts here ---------------------------------

    //  // DG3 data
    //TODO: take parameters form different file
    // float nt = 3;
    // float is_stationary = 0;
    // float gsize = 100;
    // float num_actions = 8;
    // float num_rzns = 5000;
    // float bDimx = num_rzns;
    // float F = 20.2;
    // float r_outbound = -1;
    // float r_terminal = 1;
    // float i_term = 19;
    // float j_term = 40;
    // float nmodes = 5;
    // float x0 = 0.005;
    // float y0 = 0.005;
    // float dx = 0.01; float dy = 0.01;
    // float dt = 0.0004;
    // if (num_rzns >= 1000)
    //     bDimx = 1000;

    // //TODO: define output file savepath

    // float z = -9999;
    // // TODO: 1. read paths form file
    // //       2. Make sure files are stored in np.float32 format
    // std::string data_path = "data/nT_60/";
    // std::string all_u_fname = data_path + "all_u_mat.npy";
    // std::string all_v_fname = data_path + "all_v_mat.npy";
    // std::string all_ui_fname = data_path + "all_ui_mat.npy";
    // std::string all_vi_fname = data_path + "all_vi_mat.npy";
    // std::string all_yi_fname = data_path + "all_Yi.npy";

    // simple grid
    float nt = 3;
    float is_stationary = 0;
    float gsize = 5;
    float num_actions = 8;
    float num_rzns = 5;
    float bDimx = num_rzns;
    float F = 1;
    float r_outbound = -1;
    float r_terminal = 1;
    float i_term = 1;
    float j_term = 3;
    float nmodes = 1;
    float x0 = 0.5;
    float y0 = 0.5;
    float dx = 1; float dy = 1;
    float dt = 1;
    if (num_rzns >= 1000)
        bDimx = 1000;
 
    float z = -9999;
    // TODO: 1. read paths form file
    //       2. Make sure files are stored in np.float32 format
    std::string data_path = "data/jet_5x5/";
    std::string all_u_fname = data_path + "all_u_mat.npy";
    std::string all_v_fname = data_path + "all_v_mat.npy";
    std::string all_ui_fname = data_path + "all_ui_mat.npy";
    std::string all_vi_fname = data_path + "all_vi_mat.npy";
    std::string all_yi_fname = data_path + "all_Yi.npy";


// -------------------- input data ends here ---------------------------------


    int all_u_n_elms;
    int all_v_n_elms;
    int all_ui_n_elms;
    int all_vi_n_elms;
    int all_yi_n_elms;

    cnpy::NpyArray all_u_cnpy = read_velocity_field_data(all_u_fname, &all_u_n_elms);
    cnpy::NpyArray all_v_cnpy = read_velocity_field_data(all_v_fname, &all_v_n_elms);
    cnpy::NpyArray all_ui_cnpy = read_velocity_field_data(all_ui_fname, &all_ui_n_elms);
    cnpy::NpyArray all_vi_cnpy = read_velocity_field_data(all_vi_fname, &all_vi_n_elms);
    cnpy::NpyArray all_yi_cnpy = read_velocity_field_data(all_yi_fname, &all_yi_n_elms);

    float* all_u_mat = all_u_cnpy.data<float>();
    float* all_v_mat = all_v_cnpy.data<float>();
    float* all_ui_mat = all_ui_cnpy.data<float>();
    float* all_vi_mat = all_vi_cnpy.data<float>();
    float* all_yi_mat = all_yi_cnpy.data<float>();

    std::cout << "Finished reading Velocity Field Data !" << std::endl;


    //TODO: fill params in a function
    thrust::host_vector<float> H_params(32);
    H_params[0] = gsize;
    H_params[1] = num_actions; 
    H_params[2] = num_rzns;
    H_params[3] = F;
    H_params[4] = dt;
    H_params[5] = r_outbound;
    H_params[6] = r_terminal;
    H_params[7] = nmodes;
    H_params[8] = i_term;
    H_params[9] = j_term;
    H_params[10] = nt;
    H_params[11] = is_stationary;
    for( int i =12; i<32; i++)
        H_params[i] = z;

    // Define grid ticks in host
    thrust::host_vector<float> H_xs(gsize, -1);
    thrust::host_vector<float> H_ys(gsize, -1);
    float* xs = thrust::raw_pointer_cast(&H_xs[0]);
    float* ys = thrust::raw_pointer_cast(&H_ys[0]);
    //TODO:  2. move the fucntion to a separate file
    define_xs_or_ys(xs, dx, x0, gsize);
    define_xs_or_ys(ys, dy, y0, gsize);

    // define angles in host
    thrust::host_vector<float> H_ac_angles(num_actions);
    float* ac_angles = thrust::raw_pointer_cast(&H_ac_angles[0]);
    //TODO: move to custom functions
    populate_ac_angles(ac_angles, num_actions);


    //----- start copying data to device --------

    // Copy vel field data to device memory using thrust device_vector
    thrust::device_vector<float> D_all_u_vec (all_u_mat, all_u_mat + all_u_n_elms);
    thrust::device_vector<float> D_all_v_vec (all_v_mat, all_v_mat + all_v_n_elms);
    thrust::device_vector<float> D_all_ui_vec (all_ui_mat, all_ui_mat + all_ui_n_elms);
    thrust::device_vector<float> D_all_vi_vec (all_vi_mat, all_vi_mat + all_vi_n_elms);
    thrust::device_vector<float> D_all_yi_vec (all_yi_mat, all_yi_mat + all_yi_n_elms);

    float* D_all_u_arr = thrust::raw_pointer_cast(&D_all_u_vec[0]);
    float* D_all_v_arr = thrust::raw_pointer_cast(&D_all_v_vec[0]);
    float* D_all_ui_arr = thrust::raw_pointer_cast(&D_all_ui_vec[0]);
    float* D_all_vi_arr = thrust::raw_pointer_cast(&D_all_vi_vec[0]);
    float* D_all_yi_arr = thrust::raw_pointer_cast(&D_all_yi_vec[0]);

    std::cout << "Copied to Device : Velocity Field Data !" << std::endl;

    thrust::device_vector<float> D_tdummy(2,0);
    // initialise empty device vectors. These contain time-invariant data
    thrust::device_vector<float> D_ac_angles(num_actions);
    thrust::device_vector<float> D_params(32);
    thrust::device_vector<float> D_xs(gsize);
    thrust::device_vector<float> D_ys(gsize);
    // copy data from host to device
    D_ac_angles = H_ac_angles;
    D_params = H_params;
    D_xs = H_xs;
    D_ys = H_ys;

    for(int t = 0; t < nt; t++){
        std::cout << "*** Computing data for timestep, T = " << t << std::endl;
        D_tdummy[0] = t;

        // TODO: Define output data
        build_sparse_transition_model_at_T(t, bDimx, D_tdummy, D_all_u_arr, D_all_v_arr 
                                             ,D_all_ui_arr, D_all_vi_arr, D_all_yi_arr,
                                             D_params, D_xs, D_ys, D_ac_angles);
                                            //  output_data )
        std::cout << std::endl << std::endl;
        // TODO: concatenate output data
        // concatenate_results_across_time(output_data)
    }

    return 0;
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
    std::cout << std::endl << "First 10 elements of loaded array are: " << std::endl;
    for (int i = 0; i < 10; i++)
         std::cout << vel_data[i] << "  " ;
    
    std::cout << std::endl;

    return arr;

}

void define_xs_or_ys(float* xs, float dx, float x0, int gsize){

    for(int i = 0; i < gsize;  i++)
        xs[i] = x0 + i*dx;
}


void populate_ac_angles(float* ac_angles, int num_actions){
    //fills array with equally spaced angles in radians
    for (int i = 0; i < num_actions; i++)
        ac_angles[i] = i*(2*M_PI)/num_actions;

}

// -L/usr/local/ -lcnpy -lz --std=c++11 