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



/*******************************************************************************
The following code is a modification of spvi.cu (GPU accelerated Value-Iteration
with sparse STMs)in the GEMBench project that can be found at
https://github.com/asapio/gembench

As requetsted by the authors of GEMBench, the copyright is pasted above.
*******************************************************************************/
#include <cuda_runtime.h>  // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>      // cusparseSpMV
#include <stdio.h>         // printf
#include <stdlib.h>        // EXIT_FAILURE
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <assert.h>
#include <chrono>
using namespace std::chrono;


//TODO: initialise dev_Q
static float s_discount_factor = 1; // get from file

static int*   s_dev_cooRowIndex;
static int*   s_dev_cooColIndex;
static float* s_dev_cooVal;
static float* s_dev_PV;
static float* s_dev_CV;
static int*   s_dev_CP;
static float* s_dev_Q;
static float* s_dev_R;

static cusparseSpMatDescr_t s_stms_descr;
static cusparseDnVecDescr_t vecX, vecY;

void get_MDP_model_test(){
    // output: s_stms_descr - transition matrix descriptor updated in COO format
    //         s_dev_R - reward for each state concatenated across actions. 
    const int num_actions;
    const int ncells = 4;
    const int nt = 3;
    const int nu = 2;
    const int A_num_rows = ncells * nt;
    const int A_num_cols = ncells * nt;
    const int A_num_nnz  = nu * ncells * (nt - 1);
    float alpha = 1.0f;
    float beta  = 0.0f;
    int col_st;

    thrust::host_vector<int> hA_csrOffsets_vec( A_num_rows + 1);
    thrust::host_vector<int> hA_cooRows_vec( A_num_nnz);
    thrust::host_vector<int> hA_columns_vec( A_num_nnz);
    thrust::host_vector<float> hA_values_vec( A_num_nnz);
    thrust::host_vector<float> hX_vec(A_num_cols, 1);
    thrust::host_vector<float> hY_vec(A_num_rows, 0);

    for(int i = 0; i < A_num_nnz; i++){
        hA_cooRows_vec[i] = i/nu;

        col_st = ((i/(ncells*nu)) + 1)*ncells;
        hA_columns_vec[i] = col_st + (i%nu);

        hA_values_vec[i] = (i/nu) + 1;
    }
    std::cout << std::endl;
    // //  check coo intilisations
    // for(int i = 0; i < A_num_nnz; i++)
    //     std::cout << hA_cooRows_vec[i] << ", " ;
    // std::cout << std::endl;
    // for(int i = 0; i < A_num_nnz; i++)
    //     std::cout << hA_columns_vec[i] << ", " ;   
    // std::cout << std::endl; 
    // for(int i = 0; i < A_num_nnz; i++)
    //     std::cout << hA_values_vec[i] << ", " ;  
    // std::cout << std::endl;  

    // copy COO data from host to device. 
    // TODO: 1. modify on host
    //      2. also copy prefix sum required for modification in pt1.
    thrust::device_vector<int> dA_csrOffsets_vec(A_num_rows + 1);
    thrust::device_vector<int> dA_cooRows_vec(hA_cooRows_vec.begin(), hA_cooRows_vec.end());
    thrust::device_vector<int> dA_columns_vec(hA_columns_vec.begin(), hA_columns_vec.end());
    std::cout << "flag" << std::endl;
    thrust::device_vector<float> dA_values_vec(hA_values_vec.begin(), hA_values_vec.end());
    // thrust::device_vector<float> dX_vec(hX_vec.begin(), hX_vec.end());
    // thrust::device_vector<float> dY_vec(hY_vec.begin(), hY_vec.end());

    s_dev_cooRowIndex = thrust::raw_pointer_cast(&dA_cooRows_vec[0]);
    s_dev_cooColIndex = thrust::raw_pointer_cast(&dA_columns_vec[0]);
    s_dev_cooVal = thrust::raw_pointer_cast(&dA_values_vec[0]);
    // float* dX = thrust::raw_pointer_cast(&dX_vec[0]);
    // float* dY = thrust::raw_pointer_cast(&dY_vec[0]);

    // Update global COO descriptor
    CHECK_CUSPARSE( cusparseCreateCoo(&s_stms_descr, A_num_rows, A_num_cols, A_num_nnz,
         s_dev_cooRowIndex, s_dev_cooColIndex, s_dev_cooVal,
         CUSPARSE_INDEX_32I,
         CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

    

}




static void solver_do_backup(
    const float* dev_R,
    const float* dev_PV,
    float* dev_CV,
    int* dev_CP,
    float* dev_Q)
{

    static const float fOne = 1.0f;
    cudaError_t cudaErr;

    // Copy dev_R into dev_Q
    cudaErr = cudaMemcpy(dev_Q, dev_R, (size_t)(s_NsNa*sizeof(float)), cudaMemcpyDeviceToDevice);
    assert(cudaErr == cudaSuccess);

    float alpha = s_discount_factor;

    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                        handle, 
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha,
                        s_stms_descr, 
                        &dev_PV[0], 
                        &fOne, 
                        &dev_Q[0], 
                        CUDA_R_32F,
                        CUSPARSE_MV_ALG_DEFAULT, 
                        &bufferSize) 
                    )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    std::cout << "bufferSize = " << bufferSize << std::endl;

    // execute SpMV
    CHECK_CUSPARSE( cusparseSpMV(
                        handle, 
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, 
                        s_stms_descr, 
                        vecX, 
                        &beta, 
                        vecY, 
                        CUDA_R_32F,
                        CUSPARSE_MV_ALG_DEFAULT, 
                        dBuffer) 
                    )







    // Select best action using CUDA kernel
    // Launch 1 kernel per MDP state
    // Use thread blocks with 256 threads per thread block
    // select_best_action<<<(s_Ns+255)/256, 256>>>(s_Ns, s_Na, dev_Q, dev_CV,dev_CP);

    cudaDeviceSynchronize();
}





// int solver_spvi_solve(float* s_dev_R, uint32_t* p_out_policy, float* p_out_value_func, int max_solver_time_s)
int solver_spvi_solve()
{
    printf("Solver spvi\n");

    assert(cuda_init(0) == EXIT_SUCCESS);   // TODO: check what this means exactly

    // Load in MDP from external format
    // change_mdp_format(p_mdp_obj); // we already have vectors and sparse COO  matrix
    // TODO: function to COO to CSR if necessary.

    // initilaise device vectors
    thrust::device_vector<float> s_dev_PV_vec();
    thrust::device_vector<float> s_dev_CV_vec();
    thrust::device_vector<int> s_dev_CP_vec(); // TODO: int or int32_t check.
    thrust::device_vector<float> s_dev_Q_vec();

    float* s_dev_PV = thrust::raw_pointer_cast(&s_dev_PV_vec[0]);
    float* s_dev_CV = thrust::raw_pointer_cast(&s_dev_CV_vec[0]);
    int* s_dev_CP = thrust::raw_pointer_cast(&s_dev_CP_vec[0]); // TODO: int or int32_t
    float* s_dev_Q = thrust::raw_pointer_cast(&s_dev_Q_vec[0]);


    // printf("Starting Value Iteration\n");

    struct timespec start_time, elapsed_time;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start_time);

    bool b_done = false;
    uint32_t num_iterations = 0;
    bool b_timed_out = false;
    while(!b_done)
    {
        num_iterations++;
        solver_do_backup(
                s_dev_R,
                s_dev_PV,
                s_dev_CV,
                s_dev_CP,
                s_dev_Q);

        b_done = true;

        // // Compute stopping criteria
        // float sup_norm = compute_sup_norm((const float*)s_dev_CV, (const float*)s_dev_PV, (uint32_t)s_Ns);

        // if (sup_norm < s_stopping_thresh)
        // {
        //     // Done
        //     b_done = true;
        //     printf("Iteration %d: %f < %f (STOP)\n", num_iterations, sup_norm, s_stopping_thresh);
        // }

        // // REMOVED: check for time-out by setting b_timed_out = b_done = true
        // //  TODO: perhaps add back later

        // // The value function computed in this iteration now becomes the "previous" value function.
        // cudaError_t cudaErr;
        // cudaErr = cudaMemcpy(s_dev_PV, s_dev_CV, (size_t)(s_Ns*sizeof(float)), cudaMemcpyDeviceToDevice);
        // assert(cudaErr == cudaSuccess);
    }

    // // Done. Save off policy and value
    // cudaError_t cudaErr;
    // cudaErr = cudaMemcpy(p_out_policy, s_dev_CP, (size_t)(s_Ns*sizeof(int)), cudaMemcpyDeviceToHost);
    // assert(cudaErr == cudaSuccess);

    // cudaErr = cudaMemcpy(p_out_value_func, s_dev_CV, (size_t)(s_Ns*sizeof(float)), cudaMemcpyDeviceToHost);
    // assert(cudaErr == cudaSuccess);


    // // Free any CPU RAM that was malloc'd in this function
    // if (s_host_cooRowIndex != NULL) {free(s_host_cooRowIndex);}
    // if (s_host_cooColIndex != NULL) {free(s_host_cooColIndex);}
    // if (s_host_cooVal != NULL) {free(s_host_cooVal);}
    // if (s_h_reduce_out_vec != NULL) {free(s_h_reduce_out_vec);}


    // Free all GPU memory allocations
    assert(cuda_deinit() == EXIT_SUCCESS);

    if (b_timed_out)
    {
        return(1);
    }
    else
    {
        return(0);
    }

}