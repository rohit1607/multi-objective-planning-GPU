// *** spmv_example.c ***
// How to compile (assume CUDA is installed at /usr/local/cuda/)
//   nvcc spmv_example.c -o spmv_example -L/usr/local/cuda/lib64 -lcusparse -lcudart
// or, for C compiler
//   cc -I/usr/local/cuda/include -c spmv_example.c -o spmv_example.o -std=c99
//   nvcc -lcusparse -lcudart spmv_example.o -o spmv_example
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


#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}






int main(){

    // ------------- CHECK WHETHER SPMV(COO) AND SPMV(CSR) GIVE SAME RESULTS WITH OWN MATRIX-------------------
    //  _______ both result
    // Host problem definition
    // const int A_num_rows = 4;
    // const int A_num_cols = 4;
    // const int A_num_nnz  = 9;
    // int   hA_csrOffsets[] = { 0, 3, 4, 7, 9 };
    // int   hA_columns[]    = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
    // float hA_values[]     = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
    //                             6.0f, 7.0f, 8.0f, 9.0f };
    // float hX[]            = { 1.0f, 2.0f, 3.0f, 4.0f };
    // const float result[]  = { 19.0f, 8.0f, 51.0f, 52.0f };
    // float alpha = 1.0f;
    // float beta  = 0.0f;



    const int ncells = 16*1e4;
    const int nt = 100;
    const int nu = 20;
    const int A_num_rows = ncells * nt;
    const int A_num_cols = ncells * nt;
    const int A_num_nnz  = nu * ncells * (nt - 1);
    float alpha = 1.0f;
    float beta  = 0.0f;

    thrust::host_vector<int> hA_csrOffsets_vec( A_num_rows + 1);
    thrust::host_vector<int> hA_cooRows_vec( A_num_nnz);
    thrust::host_vector<int> hA_columns_vec( A_num_nnz);
    thrust::host_vector<float> hA_values_vec( A_num_nnz);
    thrust::host_vector<float> hX_vec(A_num_cols, 1);
    thrust::host_vector<float> hY_vec(A_num_rows, 0);


    int col_st;
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



    thrust::device_vector<int> dA_csrOffsets_vec(A_num_rows + 1);
    thrust::device_vector<int> dA_cooRows_vec(hA_cooRows_vec.begin(), hA_cooRows_vec.end());
    thrust::device_vector<int> dA_columns_vec(hA_columns_vec.begin(), hA_columns_vec.end());
    std::cout << "flag" << std::endl;
    thrust::device_vector<float> dA_values_vec(hA_values_vec.begin(), hA_values_vec.end());
    thrust::device_vector<float> dX_vec(hX_vec.begin(), hX_vec.end());
    thrust::device_vector<float> dY_vec(hY_vec.begin(), hY_vec.end());



    int* dA_csrOffsets = thrust::raw_pointer_cast(&dA_csrOffsets_vec[0]);
    int* dA_columns = thrust::raw_pointer_cast(&dA_columns_vec[0]);
    float* dA_values = thrust::raw_pointer_cast(&dA_values_vec[0]);
    float* dX = thrust::raw_pointer_cast(&dX_vec[0]);
    float* dY = thrust::raw_pointer_cast(&dY_vec[0]);

    int* dA_cooRows = thrust::raw_pointer_cast(&dA_cooRows_vec[0]);


    cusparseHandle_t     handle = 0;
    cusparseStatus_t status = cusparseCreate(&handle);
    status = cusparseXcoo2csr(handle, dA_cooRows, A_num_nnz, A_num_rows, dA_csrOffsets, CUSPARSE_INDEX_BASE_ZERO);
    assert( status == CUSPARSE_STATUS_SUCCESS);
    // // check
    // std::cout<<std::endl;
    // for (int i = 0; i < A_num_rows + 1; i++)
    //     std::cout << dA_csrOffsets_vec[i] << " ";
    // std::cout<<std::endl;

    auto start = high_resolution_clock::now();

    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void*  dBuffer    = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )


    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_num_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

    // CHECK_CUSPARSE( cusparseCreateCoo(&matA, A_num_rows, A_num_cols, A_num_nnz,
    //                                  dA_cooRows, dA_columns, dA_values,
    //                                   CUSPARSE_INDEX_32I,
    //                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

    // Create dense vector X
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_32F) )
    // Create dense vector y
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_32F) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                 handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                 CUSPARSE_MV_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    std::cout << "bufferSize = " << bufferSize << std::endl;

    // execute SpMV
    CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                 CUSPARSE_MV_ALG_DEFAULT, dBuffer) )

    auto end = high_resolution_clock::now();


    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )


    auto duration = duration_cast<microseconds>(end - start);
    std::cout << "spmv = "<< duration.count()/1e6 << std::endl;

// //         // device result check
//     hY_vec = dY_vec;
   
//     std::cout << " Y vector" << std::endl;
//     for (int i = 0; i < A_num_rows; i++)
//         std::cout << hY_vec[i]<< std::endl;


    return 0;
}



// int main(){

//     // ------------- CHECK WHETHER SPMV(COO) AND SPMV(CSR) GIVE SAME RESULTS WITH DOC EXAMPLE: SUCCESS-------------------
//     // Host problem definition
//     const int A_num_rows = 4;
//     const int A_num_cols = 4;
//     const int A_num_nnz  = 9;
//     int   hA_csrOffsets[] = { 0, 3, 4, 7, 9 };
//     int   hA_columns[]    = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
//     float hA_values[]     = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
//                                 6.0f, 7.0f, 8.0f, 9.0f };
//     float hX[]            = { 1.0f, 2.0f, 3.0f, 4.0f };
//     const float result[]  = { 19.0f, 8.0f, 51.0f, 52.0f };
//     float alpha = 1.0f;
//     float beta  = 0.0f;


//     thrust::device_vector<int> dA_csrOffsets_vec(hA_csrOffsets, hA_csrOffsets + 5);
//     thrust::device_vector<int> dA_columns_vec(hA_columns, hA_columns + A_num_nnz);
//     thrust::device_vector<float> dA_values_vec(hA_values, hA_values + A_num_nnz);
//     thrust::device_vector<float> dX_vec(hX, hX + A_num_rows);
//     thrust::device_vector<float> dY_vec(A_num_cols, 0);

//     thrust::device_vector<int> dA_cooRows_vec(A_num_nnz);


//     int* dA_csrOffsets = thrust::raw_pointer_cast(&dA_csrOffsets_vec[0]);
//     int* dA_columns = thrust::raw_pointer_cast(&dA_columns_vec[0]);
//     float* dA_values = thrust::raw_pointer_cast(&dA_values_vec[0]);
//     float* dX = thrust::raw_pointer_cast(&dX_vec[0]);
//     float* dY = thrust::raw_pointer_cast(&dY_vec[0]);

//     int* dA_cooRows = thrust::raw_pointer_cast(&dA_cooRows_vec[0]);




//     // cusparseXcsr2coo(cusparseHandle_t    handle,
// //     const int*          csrRowPtr,
// //     int                 nnz,
// //     int                 m,
// //     int*                cooRowInd,
// //     cusparseIndexBase_t idxBase)

//     cusparseHandle_t     handle = 0;
//     cusparseStatus_t status = cusparseCreate(&handle);


//     status = cusparseXcsr2coo(handle, dA_csrOffsets, A_num_nnz, A_num_rows, dA_cooRows, CUSPARSE_INDEX_BASE_ZERO);
//     assert( status == CUSPARSE_STATUS_SUCCESS);
//     // check
//     std::cout<<std::endl;
//     for (int i = 0; i < A_num_nnz; i++)
//         std::cout << dA_cooRows_vec[i] << " ";
//     std::cout<<std::endl;

//     cusparseSpMatDescr_t matA;
//     cusparseDnVecDescr_t vecX, vecY;
//     void*  dBuffer    = NULL;
//     size_t bufferSize = 0;
//     CHECK_CUSPARSE( cusparseCreate(&handle) )


//     // // Create sparse matrix A in CSR format
//     // CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_num_nnz,
//     //                                   dA_csrOffsets, dA_columns, dA_values,
//     //                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
//     //                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

//     CHECK_CUSPARSE( cusparseCreateCoo(&matA, A_num_rows, A_num_cols, A_num_nnz,
//                                      dA_cooRows, dA_columns, dA_values,
//                                       CUSPARSE_INDEX_32I,
//                                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

//     // Create dense vector X
//     CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_32F) )
//     // Create dense vector y
//     CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_32F) )
//     // allocate an external buffer if needed
//     CHECK_CUSPARSE( cusparseSpMV_bufferSize(
//                                  handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//                                  &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
//                                  CUSPARSE_MV_ALG_DEFAULT, &bufferSize) )
//     CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

//     // execute SpMV
//     CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//                                  &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
//                                  CUSPARSE_MV_ALG_DEFAULT, dBuffer) )

//     // destroy matrix/vector descriptors
//     CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
//     CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
//     CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
//     CHECK_CUSPARSE( cusparseDestroy(handle) )



//         // device result check
//     // CHECK_CUDA( cudaMemcpy(hY, dY, ns * sizeof(float),
//     //                        cudaMemcpyDeviceToHost) )

//     thrust::host_vector<float> hY_vec(A_num_cols);
//     hY_vec = dY_vec;
//     int correct = 1;
//     for (int i = 0; i < A_num_rows; i++) {
//         if (hY_vec[i] != result[i]) {
//             correct = 0;
//             break;
//         }
//     }
//     if (correct)
//         printf("spmv_example test PASSED\n");
//     else
//         printf("spmv_example test FAILED: wrong result\n");

//     return 0;
// }





// int main() {
//     // Host problem definition
//     const int A_num_rows = 4;
//     const int A_num_cols = 4;
//     const int A_num_nnz  = 9;
//     int   hA_csrOffsets[] = { 0, 3, 4, 7, 9 };
//     int   hA_columns[]    = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
//     float hA_values[]     = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
//                               6.0f, 7.0f, 8.0f, 9.0f };
//     float hX[]            = { 1.0f, 2.0f, 3.0f, 4.0f };
//     const float result[]  = { 19.0f, 8.0f, 51.0f, 52.0f };
//     float alpha = 1.0f;
//     float beta  = 0.0f;

    
//     // coo intialisation
//     const int nc = 4;
//     const int nt = 4;
//     const int ns = nc*nt;
//     const int nu = 2;
//     const int nnz = nu * nc * (nt - 1);
//     int col_st;

//     int* hA_coo_rowId = new int[nnz];
//     int* hA_coo_colId = new int[nnz];
//     int* hA_coo_Val = new int[nnz];
//     float* hX2 = new float[ns];

//     for( int i=0; i<ns; i++)
//         hX2[i] = 1.0f;

//     for(int i = 0; i < nnz; i++){
//         hA_coo_rowId[i] = i/nu;

//         col_st = ((i/(nc*nu)) + 1)*nc;
//         hA_coo_colId[i] = col_st + (i%nu);

//         hA_coo_Val[i] = (i/nu) + 1;
//     }
//     std::cout << std::endl;
//     // //  check coo intilisations
//     // for(int i = 0; i < nnz; i++)
//     //     std::cout << hA_coo_rowId[i] << ", " ;
//     // std::cout << std::endl;
//     // for(int i = 0; i < nnz; i++)
//     //     std::cout << hA_coo_colId[i] << ", " ;   
//     // std::cout << std::endl; 
//     // for(int i = 0; i < nnz; i++)
//     //     std::cout << hA_coo_Val[i] << ", " ;  
//     // std::cout << std::endl;  
    
//     //--------------------------------------------------------------------------
//     // Device memory management
//     int   *dA_csrOffsets, *dA_columns, *dA_coo_colId, *dA_coo_rowId;
//     float *dA_values, *dX, *dY, *dA_coo_Val, *dX2, *dY2;

//     CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
//                            (A_num_rows + 1) * sizeof(int)) )
//     CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_num_nnz * sizeof(int))    )
//     CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_num_nnz * sizeof(float))  )
//     CHECK_CUDA( cudaMalloc((void**) &dX,         A_num_cols * sizeof(float)) )
//     CHECK_CUDA( cudaMalloc((void**) &dY,         A_num_rows * sizeof(float)) )

//     CHECK_CUDA( cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
//                            (A_num_rows + 1) * sizeof(int),
//                            cudaMemcpyHostToDevice) )
//     CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_num_nnz * sizeof(int),
//                            cudaMemcpyHostToDevice) )
//     CHECK_CUDA( cudaMemcpy(dA_values, hA_values,
//                            A_num_nnz * sizeof(float), cudaMemcpyHostToDevice) )
//     CHECK_CUDA( cudaMemcpy(dX, hX, A_num_rows * sizeof(float),
//                            cudaMemcpyHostToDevice) )


//     CHECK_CUDA( cudaMalloc((void**) &dA_coo_rowId,
//                            (nnz) * sizeof(int)) )
//     CHECK_CUDA( cudaMalloc((void**) &dA_coo_colId, nnz * sizeof(int))    )
//     CHECK_CUDA( cudaMalloc((void**) &dA_coo_Val,  nnz * sizeof(float))  )
//     CHECK_CUDA( cudaMalloc((void**) &dX2,         ns * sizeof(float)) )
//     CHECK_CUDA( cudaMalloc((void**) &dY2,         ns * sizeof(float)) )

//     CHECK_CUDA( cudaMemcpy(dA_coo_rowId, hA_coo_rowId,
//                            (nnz) * sizeof(int),
//                            cudaMemcpyHostToDevice) )
//     CHECK_CUDA( cudaMemcpy(dA_coo_colId, hA_coo_colId, nnz * sizeof(int),
//                            cudaMemcpyHostToDevice) )
//     CHECK_CUDA( cudaMemcpy(dA_coo_Val, hA_coo_Val,
//                            nnz * sizeof(float), cudaMemcpyHostToDevice) )
//     CHECK_CUDA( cudaMemcpy(dX2, hX2, ns * sizeof(float),
//                            cudaMemcpyHostToDevice) )
//     //--------------------------------------------------------------------------
//     // CUSPARSE APIs
//     cusparseHandle_t     handle = 0;

//     cusparseSpMatDescr_t matA;
//     cusparseDnVecDescr_t vecX, vecY;
//     void*  dBuffer    = NULL;
//     size_t bufferSize = 0;
//     CHECK_CUSPARSE( cusparseCreate(&handle) )
//     // Create sparse matrix A in COO format
//     // cusparseCreateCoo(cusparseSpMatDescr_t* spMatDescr,
//     //     int64_t               rows,
//     //     int64_t               cols,
//     //     int64_t               nnz,
//     //     void*                 cooRowInd,
//     //     void*                 cooColInd,
//     //     void*                 cooValues,
//     //     cusparseIndexType_t   cooIdxType,
//     //     cusparseIndexBase_t   idxBase,
//     //     cudaDataType          valueType)
//     CHECK_CUSPARSE( cusparseCreateCoo(&matA, ns, ns, nnz,
//                                       dA_coo_rowId, dA_coo_colId, dA_coo_Val,
//                                       CUSPARSE_INDEX_32I,
//                                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
//     // Create dense vector X
//     CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, ns, dX2, CUDA_R_32F) )
//     // Create dense vector y
//     CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, ns, dY2, CUDA_R_32F) )
//     // allocate an external buffer if needed
//     CHECK_CUSPARSE( cusparseSpMV_bufferSize(
//                                  handle,   CUSPARSE_OPERATION_NON_TRANSPOSE ,
//                                  &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
//                                  CUSPARSE_MV_ALG_DEFAULT, &bufferSize) )
//     CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

//     // execute SpMV
//     CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//                                  &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
//                                  CUSPARSE_MV_ALG_DEFAULT, dBuffer) )

//     // destroy matrix/vector descriptors
//     CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
//     CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
//     CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
//     CHECK_CUSPARSE( cusparseDestroy(handle) )

//     // cusparseSpMatDescr_t matA;
//     // cusparseDnVecDescr_t vecX, vecY;
//     // void*  dBuffer    = NULL;
//     // size_t bufferSize = 0;
//     // CHECK_CUSPARSE( cusparseCreate(&handle) )
//     // // Create sparse matrix A in CSR format
//     // CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_num_nnz,
//     //                                   dA_csrOffsets, dA_columns, dA_values,
//     //                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
//     //                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
//     // // Create dense vector X
//     // CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_32F) )
//     // // Create dense vector y
//     // CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_32F) )
//     // // allocate an external buffer if needed
//     // CHECK_CUSPARSE( cusparseSpMV_bufferSize(
//     //                              handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//     //                              &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
//     //                              CUSPARSE_MV_ALG_DEFAULT, &bufferSize) )
//     // CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

//     // // execute SpMV
//     // CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//     //                              &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
//     //                              CUSPARSE_MV_ALG_DEFAULT, dBuffer) )

//     // // destroy matrix/vector descriptors
//     // CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
//     // CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
//     // CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
//     // CHECK_CUSPARSE( cusparseDestroy(handle) )
//     //--------------------------------------------------------------------------
//     // device result check
//     float hY2[ns];
//     CHECK_CUDA( cudaMemcpy(hY2, dY2, ns * sizeof(float),
//                            cudaMemcpyDeviceToHost) )

//     std::cout << std::endl;
//     for ( int i = 0; i < ns; i++)
//         std::cout << hY2[i] << std::endl;

//     // int correct = 1;
//     // for (int i = 0; i < A_num_rows; i++) {
//     //     if (hY[i] != result[i]) {
//     //         correct = 0;
//     //         break;
//     //     }
//     // }
//     // if (correct)
//     //     printf("spmv_example test PASSED\n");
//     // else
//     //     printf("spmv_example test FAILED: wrong result\n");
//     //--------------------------------------------------------------------------
//     // device memory deallocation
//     CHECK_CUDA( cudaFree(dBuffer) )
//     CHECK_CUDA( cudaFree(dA_csrOffsets) )
//     CHECK_CUDA( cudaFree(dA_columns) )
//     CHECK_CUDA( cudaFree(dA_values) )
//     CHECK_CUDA( cudaFree(dX) )
//     CHECK_CUDA( cudaFree(dY) )
//     return EXIT_SUCCESS;
//     // return 0;
// }
