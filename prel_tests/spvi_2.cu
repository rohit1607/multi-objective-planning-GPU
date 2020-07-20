/*******************************************************************************
@ddblock_begin copyright

Copyright (c) 1997-2019
Maryland DSPCAD Research Group, The University of Maryland at College Park 

Permission is hereby granted, without written agreement and without license or
royalty fees, to use, copy, modify, and distribute this software and its
documentation for any purpose other than its incorporation into a commercial
product, provided that the above copyright notice and the following two
paragraphs appear in all copies of this software.

IN NO EVENT SHALL THE UNIVERSITY OF MARYLAND BE LIABLE TO ANY PARTY
FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF
THE UNIVERSITY OF MARYLAND HAS BEEN ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.

THE UNIVERSITY OF MARYLAND SPECIFICALLY DISCLAIMS ANY WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE
PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE UNIVERSITY OF
MARYLAND HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
ENHANCEMENTS, OR MODIFICATIONS.

@ddblock_end copyright
*******************************************************************************/

// C
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <chrono>
#include "cnpy.h"


using namespace std::chrono;
// CUDA Runtime
#include <cuda.h>
#include <cuda_runtime.h>
#include "cusparse.h"

// // Solver interfaces
// #include "solver_spvi.h"



static int    s_nnz = 0;
static size_t s_Ns = 0;
static size_t s_Na = 0;
static size_t s_NsNa = 0;     // Shorthand for "Ns times Na"
static size_t s_Ns2Na = 0;    // Shorthand for "Ns squared times Na"
static float s_discount_factor = 1;
static float s_stopping_thresh = 0;

// Pointers to buffers in CPU RAM
static int32_t*   s_host_cooRowIndex = NULL;
static int32_t*   s_host_cooColIndex = NULL;
static float* s_host_cooVal = NULL;

// Pointers to buffers in GPU
static float* s_dev_PV;
static float* s_dev_CV;
static int32_t*   s_dev_CP;
static float* s_dev_Q;
static float* s_dev_R;
static int32_t*   s_dev_cooRowIndex;
static int32_t*   s_dev_cooColIndex;
static float* s_dev_cooVal;
static int32_t*   s_dev_csrRowPtr=0;

void*  dBuffer    = NULL;
size_t bufferSize = 0;

static cusparseHandle_t s_handle = 0;
static cusparseSpMatDescr_t s_stms_descr;


// Memory using in sup_norm reduction kernel
// Needs file scope so we can free the malloc'd memory
// after the solver completes
static float* s_h_reduce_out_vec = NULL;
static float* s_d_reduce_out_vec = NULL;

__global__
void select_best_action(int num_states, int num_actions, const float *dev_Q, float *dev_CV, int32_t* dev_CP)
{
    int n = blockIdx.x*blockDim.x + threadIdx.x;

    // More kernels than states will be launched, dont go out of bounds
    if (n < num_states)
    {
        float max_value = -1e6;
        int32_t best_action = -1;

        for (int a_idx=0; a_idx<num_actions; a_idx++)
        {
            // Compute index in Q
            int32_t q_index = a_idx*(num_states) + n;

            float value_for_this_action = dev_Q[q_index];

            // Is this the new best action?
            if (value_for_this_action > max_value)
            {
                max_value = value_for_this_action;
                best_action = a_idx;
            }
        }

        dev_CV[n] = max_value;
        dev_CP[n] = best_action;
    }
}

// Reduction kernel taken from "reduction" example in CUDA samples.
// More info can be found here:
// http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
// This is the "#4" example in the presentation

__global__
void reduce_sup_norm(const float *g_idata_1, const float *g_idata_2, float *g_odata, unsigned int n)
{
    extern __shared__ float sdata[];

//    // perform first level of reduction,
//    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

//    T mySum = (i < n) ? g_idata[i] : 0;
    float myMaxDelta = (i < n) ? fabsf(g_idata_1[i]-g_idata_2[i]) : 0.0f;

    if ((i + blockDim.x) < n)
    {
//        mySum += g_idata[i+blockDim.x];
        float newDelta = fabsf(g_idata_1[i+blockDim.x]-g_idata_2[i+blockDim.x]);
        myMaxDelta = newDelta > myMaxDelta ? newDelta : myMaxDelta;
    }

    sdata[tid] = myMaxDelta;
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
//            sdata[tid] = mySum = mySum + sdata[tid + s];
            float newDelta = sdata[tid + s];
            myMaxDelta = newDelta > myMaxDelta ? newDelta : myMaxDelta;
            sdata[tid] = myMaxDelta;
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
    {
//        g_odata[blockIdx.x] = mySum;
        g_odata[blockIdx.x] = myMaxDelta;
    }
}


static void solver_do_backup(
        const float* dev_R,
        float* dev_PV,
        float* dev_CV,
        int32_t* dev_CP,
        float* dev_Q)
{
    static const float fOne = 1.0f;

    cudaError_t cudaErr;

    // Copy dev_R into dev_Q
    cudaErr = cudaMemcpy(dev_Q, dev_R, (size_t)(s_NsNa*sizeof(float)), cudaMemcpyDeviceToDevice);
    assert(cudaErr == cudaSuccess);
    
    float alpha = s_discount_factor;

    // Multiply Matrix times vector 
    // TODO: Change function
    cusparseCreate(&s_handle);

    cusparseDnVecDescr_t vecX, vecY;
    cusparseStatus_t status;

    // Create dense vector X
    status = cusparseCreateDnVec(&vecX, s_Ns, dev_PV, CUDA_R_32F);
    assert(status == CUSPARSE_STATUS_SUCCESS);

    // Create dense vector y
    status = cusparseCreateDnVec(&vecY, s_NsNa, dev_Q, CUDA_R_32F);
    assert(status == CUSPARSE_STATUS_SUCCESS);
    
    // -----Check for correctness 
    // std::cout << "s_Ns " << s_Ns << std::endl;
    // std::cout << "s_NsNa " << s_NsNa << std::endl;

    // std::cout << "testPV[i]" << std::endl;
    // thrust::device_vector<float> testPV(dev_PV, dev_PV + s_Ns);
    // for(int i = 0; i < testPV.size(); i++)
    //     std::cout << testPV[i] << std::endl;

    // std::cout << "testQ[i]- pre mult" << std::endl;
    // thrust::device_vector<float> testQ(dev_Q, dev_Q + s_NsNa);
    // for(int i = 0; i < testQ.size(); i++)
    //     std::cout << testQ[i] << std::endl;

    // std::cout << "alpha = " << alpha << std::endl;
    // std::cout << "fOne = " << fOne << std::endl;


    // allocate an external buffer if needed
    status = cusparseSpMV_bufferSize( s_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, s_stms_descr, vecX, &fOne, vecY, CUDA_R_32F,
                            CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
    assert(status == CUSPARSE_STATUS_SUCCESS);

    cudaMalloc(&dBuffer, bufferSize);
    // execute SpMV
    status = cusparseSpMV(s_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, s_stms_descr, vecX, &fOne, vecY, CUDA_R_32F,
        CUSPARSE_MV_ALG_DEFAULT, dBuffer) ;
    assert(status == CUSPARSE_STATUS_SUCCESS);

    // check for correct multiplication
    // std::cout << "testQ[i]- post mult" << std::endl;
    // thrust::device_vector<float> test1(dev_Q, dev_Q + s_NsNa);
    // for(int i = 0; i < test1.size(); i++)
    //     std::cout << test1[i] << std::endl;
 

    // Select best action using CUDA kernel
    // Launch 1 kernel per MDP state
    // Use thread blocks with 256 threads per thread block
    select_best_action<<<(s_Ns+255)/256, 256>>>(s_Ns, s_Na, dev_Q, dev_CV, dev_CP);

    cudaDeviceSynchronize();
    // check for cactions selection kernel
    // std::cout << "testCV- post kernel in solver_do_backup" << std::endl;
    // thrust::device_vector<float> testCV(dev_CV, dev_CV + s_Ns);
    // for(int i = 0; i < testCV.size(); i++)
    //    std::cout << testCV[i] << std::endl;

    // std::cout << "testCP- post kernel in solver_do_backup" << std::endl;
    // thrust::device_vector<int32_t> testCP(dev_CP, dev_CP + s_Ns);
    // for(int i = 0; i < testCP.size(); i++)
    //    std::cout << testCP[i] << std::endl;
}


float compute_sup_norm(const float* dev_v1,
                       const float* dev_v2,
                       uint32_t N)
{
//    static int kernel_num_blocks = (N+255)/256;
    static int kernel_num_blocks = (N+255)/(256*2); // Need half the blocks due to optimization in kernel
    static int kernel_num_threads = 256;

    cudaError_t cudaErr;

    // #warning "Temp hack using CPU Sup Norm!"

    // USE CPU version for now
    if (kernel_num_blocks == 0)
    {
        // N here is <= 256
        float host_v1[N];
        float host_v2[N];

        cudaErr = cudaMemcpy(host_v1, dev_v1, (size_t)(N*sizeof(float)), cudaMemcpyDeviceToHost);
        assert(cudaErr == cudaSuccess);
        cudaErr = cudaMemcpy(host_v2, dev_v2, (size_t)(N*sizeof(float)), cudaMemcpyDeviceToHost);
        assert(cudaErr == cudaSuccess);

        float max_abs_delta = 0.0f;
        float abs_delta;
        for (uint32_t n=0; n<N; n++)
        {
            abs_delta = fabsf(host_v1[n]-host_v2[n]);
            if (abs_delta > max_abs_delta)
            {
                // printf("[%d] %f > %f, new_max_abs_delta\n", n, abs_delta, max_abs_delta);
                max_abs_delta = abs_delta;
            }
        }
        return max_abs_delta;
    }
    else
    {
        // USE GPU VERSION
        if (s_h_reduce_out_vec == NULL)
        {
            // TODO - We could allocate these earlier. Doing them here since size is a function of
            // kernel_num_blocks

            #ifdef ALLOW_PRINTS
            printf("N = %d, NB = %d, NT = %d\n", N, kernel_num_blocks, kernel_num_threads);
            #endif

            s_h_reduce_out_vec = (float*)malloc(sizeof(float)*kernel_num_blocks);
            assert(s_h_reduce_out_vec != NULL);
        }
        if (s_d_reduce_out_vec == NULL)
        {

            cudaErr = cudaMalloc((void**)&s_d_reduce_out_vec, kernel_num_blocks*sizeof(float));
            assert(cudaErr == cudaSuccess);
        }

        // Do first stage reduction using CUDA kernel
        // This leaves a length kernel_num_blocks array that needs to still be reduced
        reduce_sup_norm<<<kernel_num_blocks, kernel_num_threads, kernel_num_threads*sizeof(float)>>>(dev_v1, dev_v2, s_d_reduce_out_vec, N);
        cudaDeviceSynchronize();

        cudaErr = cudaMemcpy(s_h_reduce_out_vec, s_d_reduce_out_vec, (size_t)(kernel_num_blocks*sizeof(float)), cudaMemcpyDeviceToHost);
        // checkCudaErrors(cudaErr);
        // assert(cudaErr == cudaSuccess);

        float temp_max = 0.0f;
        for (int n=0; n<kernel_num_blocks; n++)
        {
            if (s_h_reduce_out_vec[n] > temp_max)
            {
                temp_max = s_h_reduce_out_vec[n];
            }
        }

        //printf("CPU,CUDA sup_norm = %f %f\t", max_abs_delta, temp_max);
        return temp_max;
    }

}


template<typename T>
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

    T* vel_data = arr.data<T>();
    // print check first 10 elements
    std::cout << std::endl << "First 10 elements of loaded array are: " << std::endl;
    for (int i = 0; i < 10; i++)
         std::cout << vel_data[i] << "  " ;
    
    std::cout << std::endl;

    return arr;

}

static void get_mdp_model(std::string model_fpath){

    //load npy arrays from saved files
    int dummy1, dummy2, dummy3;
    int master_nnz;
    int dummy_NsNa;
    
    cnpy::NpyArray master_cooS1_cnpy = read_velocity_field_data<int32_t>(model_fpath + "master_cooS1.npy", &master_nnz);
    cnpy::NpyArray master_cooS2_cnpy = read_velocity_field_data<int32_t>(model_fpath + "master_cooS2.npy", &dummy1);
    cnpy::NpyArray master_cooVal_cnpy = read_velocity_field_data<float>(model_fpath + "master_cooVal.npy", &dummy2);
    cnpy::NpyArray master_R_cnpy = read_velocity_field_data<float>(model_fpath + "master_R.npy", &dummy_NsNa);
    cnpy::NpyArray DP_relv_params_cnpy = read_velocity_field_data<int32_t>(model_fpath + "DP_relv_params.npy", &dummy3);
 
    int32_t* master_cooS1_arr = master_cooS1_cnpy.data<int32_t>();
    int32_t* master_cooS2_arr = master_cooS2_cnpy.data<int32_t>();
    float* master_cooVal_arr = master_cooVal_cnpy.data<float>();
    float* master_R_arr = master_R_cnpy.data<float>();
    int32_t* DP_relv_params_arr = DP_relv_params_cnpy.data<int32_t>();


    // write information to global variables.
    s_Ns = DP_relv_params_arr[0];
    s_Na = DP_relv_params_arr[1];
    s_nnz = master_nnz;
    s_NsNa = s_Ns * s_Na;
    s_Ns2Na = s_Ns*s_Ns*s_Na;


    std::cout<< "s_Ns = " << s_Ns << std::endl;
    std::cout<< "s_Na = " << s_Na << std::endl;
    std::cout<< "s_nnz = " << s_nnz << std::endl;


    //Sanity checks
    assert(dummy1 == master_nnz);
    assert(dummy2 == master_nnz);
    assert(dummy_NsNa = s_Ns*s_Na);


    //initialise host vector to allocate device memory and copy to it.
    thrust::host_vector<float> s_host_PV_vec(s_Ns, 0);
    float* s_host_PV = thrust::raw_pointer_cast(&s_host_PV_vec[0]);


    //Memrory allocations on device
    cudaError_t cudaStat;
    // Previous value function (init to zero)
    cudaStat = cudaMalloc((void**)&s_dev_PV, s_Ns*sizeof(float));
    assert(cudaStat == cudaSuccess);

    cudaStat = cudaMalloc((void**)&s_dev_CV, s_Ns*sizeof(float));
    assert(cudaStat == cudaSuccess);

    cudaStat = cudaMalloc((void**)&s_dev_CP, s_Ns*sizeof(int32_t));
    assert(cudaStat == cudaSuccess);

    cudaStat = cudaMalloc((void**)&s_dev_Q, s_NsNa*sizeof(float));
    assert(cudaStat == cudaSuccess);

    // Rewards
    cudaStat = cudaMalloc((void**)&s_dev_R, s_NsNa*sizeof(float));
    assert(cudaStat == cudaSuccess);

    // STMs
    cudaStat = cudaMalloc((void**)&s_dev_cooRowIndex, s_nnz*sizeof(int32_t));
    assert(cudaStat == cudaSuccess);

    cudaStat = cudaMalloc((void**)&s_dev_cooColIndex, s_nnz*sizeof(int32_t));
    assert(cudaStat == cudaSuccess);

    cudaStat = cudaMalloc((void**)&s_dev_cooVal, s_nnz*sizeof(float));
    assert(cudaStat == cudaSuccess);


    // -------------------------------------
    // Copy data to device
    // -------------------------------------

    cudaStat = cudaMemcpy(s_dev_PV, s_host_PV, (size_t)(s_Ns*sizeof(float)), cudaMemcpyHostToDevice);
    assert(cudaStat == cudaSuccess);

    // Copy STM from host to device
    cudaStat = cudaMemcpy(s_dev_cooRowIndex, master_cooS1_arr, (size_t)(s_nnz*sizeof(int32_t)), cudaMemcpyHostToDevice);
    assert(cudaStat == cudaSuccess);

    cudaStat = cudaMemcpy(s_dev_cooColIndex, master_cooS2_arr, (size_t)(s_nnz*sizeof(int32_t)), cudaMemcpyHostToDevice);
    assert(cudaStat == cudaSuccess);

    cudaStat = cudaMemcpy(s_dev_cooVal, master_cooVal_arr, (size_t)(s_nnz*sizeof(float)), cudaMemcpyHostToDevice);
    assert(cudaStat == cudaSuccess);

    cudaStat = cudaMemcpy(s_dev_R, master_R_arr, (size_t)(s_NsNa*sizeof(float)), cudaMemcpyHostToDevice);
    assert(cudaStat == cudaSuccess);


    // Create COO matrix
    cusparseCreate(&s_handle);
    cusparseCreateCoo(&s_stms_descr, 
                        s_NsNa,     // num of rows in concated matrix
                        s_Ns,       // num of cols in cocated matrix
                        s_nnz,      // num of nnz in concated matrix
                        s_dev_cooRowIndex, // hA
                        s_dev_cooColIndex, 
                        s_dev_cooVal,
                        CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    // Check
    // std::cout << "testPV[i]- init" << std::endl;
    // thrust::device_vector<float> testPV(s_dev_PV, s_dev_PV + s_Ns);
    // for(int i = 0; i < testPV.size(); i++)
    //     std::cout << testPV[i] << std::endl;
    
}

static void get_mdp_model_test(){
    s_discount_factor = 1;
    int s_ncells = 4;
    int s_nt =3;
    int s_nu =2;
    int s_num_actions = 2;
    int dummy_nnz =  s_ncells * s_nu * (s_nt-1) * s_num_actions;
    int nnz_per_ac = s_ncells * s_nu * (s_nt-1);
    float threshold = 1e-2;

    s_Ns = s_ncells * s_nt;
    s_Na = s_num_actions;

    s_NsNa = s_Ns*s_Na;
    s_Ns2Na = s_Ns*s_Ns*s_Na;

    // float eps = 0.5f;
    s_stopping_thresh = threshold;
    s_nnz = dummy_nnz; // cam get from master_coo host_vector.size()

    //TODO: load np array to host_vector
    thrust::host_vector<int32_t> hA_cooRows_vec( s_nnz);
    thrust::host_vector<int32_t> hA_columns_vec( s_nnz);
    thrust::host_vector<float> hA_values_vec( s_nnz);
    thrust::host_vector<float> host_R_vec(s_NsNa);
    thrust::host_vector<float> s_host_PV_vec(s_Ns, 0);
    // thrust::host_vector<float> hY_vec(s_Ns, 0);

    int32_t* s_host_cooRowIndex= thrust::raw_pointer_cast(&hA_cooRows_vec[0]);
    int32_t* s_host_cooColIndex = thrust::raw_pointer_cast(&hA_columns_vec[0]); 
    float* s_host_cooVal = thrust::raw_pointer_cast(&hA_values_vec[0]);    
    float* host_R = thrust::raw_pointer_cast(&host_R_vec[0]);
    float* s_host_PV = thrust::raw_pointer_cast(&s_host_PV_vec[0]);
    int col_st;

    for(int i = 0; i < s_nnz; i++){
        hA_cooRows_vec[i] = ((i%nnz_per_ac)/s_nu) + s_Ns*(i/nnz_per_ac);

        col_st = (((i%nnz_per_ac)/(s_ncells*s_nu)) + 1)*s_ncells;
        hA_columns_vec[i] = col_st + (i%s_nu);

        hA_values_vec[i] = 0.5f;
    }

    for(int i = 0; i < s_NsNa; i++){
        host_R[i] = 10*(i/s_Ns) + i;
    }
    std::cout << std::endl;

    //  check coo intilisations
    for(int i = 0; i < s_nnz; i++)
        std::cout << hA_cooRows_vec[i] << ", " ;
    std::cout << std::endl;
    for(int i = 0; i < s_nnz; i++)
        std::cout << hA_columns_vec[i] << ", " ;   
    std::cout << std::endl; 
    for(int i = 0; i < s_nnz; i++)
        std::cout << hA_values_vec[i] << ", " ;  
    std::cout << std::endl;  
    for(int i = 0; i < s_NsNa; i++)
        std::cout << host_R[i] << ", " ;  
    std::cout << std::endl;  


    cudaError_t cudaStat;
    // Previous value function (init to zero)
    cudaStat = cudaMalloc((void**)&s_dev_PV, s_Ns*sizeof(float));
    assert(cudaStat == cudaSuccess);

    cudaStat = cudaMalloc((void**)&s_dev_CV, s_Ns*sizeof(float));
    assert(cudaStat == cudaSuccess);

    cudaStat = cudaMalloc((void**)&s_dev_CP, s_Ns*sizeof(int32_t));
    assert(cudaStat == cudaSuccess);

    cudaStat = cudaMalloc((void**)&s_dev_Q, s_NsNa*sizeof(float));
    assert(cudaStat == cudaSuccess);

    // Rewards
    cudaStat = cudaMalloc((void**)&s_dev_R, s_NsNa*sizeof(float));
    assert(cudaStat == cudaSuccess);

    // STMs
    cudaStat = cudaMalloc((void**)&s_dev_cooRowIndex, s_nnz*sizeof(int32_t));
    assert(cudaStat == cudaSuccess);

    cudaStat = cudaMalloc((void**)&s_dev_cooColIndex, s_nnz*sizeof(int32_t));
    assert(cudaStat == cudaSuccess);

    cudaStat = cudaMalloc((void**)&s_dev_cooVal, s_nnz*sizeof(float));
    assert(cudaStat == cudaSuccess);


    // -------------------------------------
    // Copy data to device
    // -------------------------------------

    cudaStat = cudaMemcpy(s_dev_PV, s_host_PV, (size_t)(s_Ns*sizeof(float)), cudaMemcpyHostToDevice);
    assert(cudaStat == cudaSuccess);

    // Copy STM from host to device
    cudaStat = cudaMemcpy(s_dev_cooRowIndex, s_host_cooRowIndex, (size_t)(s_nnz*sizeof(int32_t)), cudaMemcpyHostToDevice);
    assert(cudaStat == cudaSuccess);

    cudaStat = cudaMemcpy(s_dev_cooColIndex, s_host_cooColIndex, (size_t)(s_nnz*sizeof(int32_t)), cudaMemcpyHostToDevice);
    assert(cudaStat == cudaSuccess);

    cudaStat = cudaMemcpy(s_dev_cooVal, s_host_cooVal, (size_t)(s_nnz*sizeof(float)), cudaMemcpyHostToDevice);
    assert(cudaStat == cudaSuccess);

    cudaStat = cudaMemcpy(s_dev_R, host_R, (size_t)(s_NsNa*sizeof(float)), cudaMemcpyHostToDevice);
    assert(cudaStat == cudaSuccess);

    cusparseCreate(&s_handle);
    cusparseCreateCoo(&s_stms_descr, 
                        s_NsNa,     // num of rows in concated matrix
                        s_Ns,       // num of cols in cocated matrix
                        s_nnz,      // num of nnz in concated matrix
                        s_dev_cooRowIndex, // hA
                        s_dev_cooColIndex, 
                        s_dev_cooVal,
                        CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    std::cout << "testPV[i]- init" << std::endl;
    thrust::device_vector<float> testPV(s_dev_PV, s_dev_PV + s_Ns);
    for(int i = 0; i < testPV.size(); i++)
        std::cout << testPV[i] << std::endl;
    

}

int solver_spvi_solve(thrust::host_vector<uint32_t>  &p_out_policy_vec, 
                        thrust::host_vector<float>  &p_out_value_func_vec, 
                        int max_solver_time_s)
{
    printf("Solver spvi\n");


    // resize host vectors for storing output
    p_out_policy_vec.resize(s_Ns);
    p_out_value_func_vec.resize(s_Ns);
    uint32_t* p_out_policy = thrust::raw_pointer_cast(&p_out_policy_vec[0]);
    float* p_out_value_func = thrust::raw_pointer_cast(&p_out_value_func_vec[0]);

    struct timespec start_time, elapsed_time;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start_time);

    bool b_done = false;
    uint32_t num_iterations = 0;
    bool b_timed_out = false;
    while(!b_done)
    {
        num_iterations++;
        std::cout << "Iter No. " << num_iterations << std::endl;
        solver_do_backup(
                s_dev_R,
                s_dev_PV,
                s_dev_CV,
                s_dev_CP,
                s_dev_Q);

        // Compute stopping criteria
        float sup_norm = compute_sup_norm((const float*)s_dev_CV, (const float*)s_dev_PV, (uint32_t)s_Ns);

        if (sup_norm < s_stopping_thresh)
        {
            // Done
            b_done = true;
            printf("Iteration %d: %f < %f (STOP)\n", num_iterations, sup_norm, s_stopping_thresh);
        }
       
        //        if (num_iterations == 2) b_done = true;

        // The value function computed in this iteration now becomes the "previous" value function.
        cudaError_t cudaErr;
        cudaErr = cudaMemcpy(s_dev_PV, s_dev_CV, (size_t)(s_Ns*sizeof(float)), cudaMemcpyDeviceToDevice);
        assert(cudaErr == cudaSuccess);
    }

    std::cout << "num_iterations" << num_iterations << std::endl;
    // Done. Save off policy and value
    cudaError_t cudaErr;
    cudaErr = cudaMemcpy(p_out_policy, s_dev_CP, (size_t)(s_Ns*sizeof(int32_t)), cudaMemcpyDeviceToHost);
    assert(cudaErr == cudaSuccess);

    cudaErr = cudaMemcpy(p_out_value_func, s_dev_CV, (size_t)(s_Ns*sizeof(float)), cudaMemcpyDeviceToHost);
    assert(cudaErr == cudaSuccess);


    // Free any CPU RAM that was malloc'd in this function
    if (s_host_cooRowIndex != NULL) {free(s_host_cooRowIndex);}
    if (s_host_cooColIndex != NULL) {free(s_host_cooColIndex);}
    if (s_host_cooVal != NULL) {free(s_host_cooVal);}
    if (s_h_reduce_out_vec != NULL) {free(s_h_reduce_out_vec);}


    // Free all GPU memory allocations
    // assert(cuda_deinit() == EXIT_SUCCESS);

    if (b_timed_out)
    {
        return(1);
    }
    else
    {
        return(0);
    }

}

int main(){

    std::string data_path = "data/output/test_DG_nt60/";
    s_stopping_thresh = 1e-1;



    int max_solver_time_s = 1;
    thrust::host_vector<uint32_t> p_out_policy_vec(0);
    thrust::host_vector<float> p_out_value_func_vec(0);
    // Load in MDP from external format
    // get_mdp_model_test();

    get_mdp_model(data_path);
    printf("mdp model laoded\n");

    solver_spvi_solve(p_out_policy_vec, p_out_value_func_vec,  max_solver_time_s);


    return 0;

}









int test_main(){
    //     // ----------------- Inidividual function Tests-----------------
    //     get_mdp_model_test();

    //     std::cout << "s_nnz = " << s_nnz << std::endl; 

    //     solver_do_backup(
    //         s_dev_R,
    //         s_dev_PV,
    //         s_dev_CV,
    //         s_dev_CP,
    //         s_dev_Q);
    //     std::cout << "s_dev_Q in main" << std::endl;
    //     thrust::device_vector<float> test(s_dev_Q, s_dev_Q + s_NsNa);
    //     for(int i = 0; i < test.size(); i++)
    //         std::cout << test[i] << std::endl;

    //     // check for cactions selection kernel
    //     std::cout << "testCV- post kernel in main" << std::endl;
    //     thrust::device_vector<float> testCV(s_dev_CV, s_dev_CV + s_Ns);
    //     for(int i = 0; i < testCV.size(); i++)
    //         std::cout << testCV[i] << std::endl;

    //     std::cout << "testPV- post kernel in main" << std::endl;
    //     thrust::device_vector<float> testPV(s_dev_PV, s_dev_PV + s_Ns);
    //     for(int i = 0; i < testPV.size(); i++)
    //         std::cout << testPV[i] << std::endl;


    //     float sup_norm = compute_sup_norm((const float*)s_dev_CV, (const float*)s_dev_PV, (uint32_t)s_Ns);
    //     std::cout << "sup_norm = " << sup_norm << std::endl;


    return 0;
}

