#include "cusparse.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

static cusparseHandle_t s_handle = 0;
static cusparseMatDescr_t s_stms_descr=0;

int nnz = 1e7;
int nrows = 1e7;
int cols = 1e7;

int main(){
    thrust::device_vector<int> s_dev_cooRowIndex(nnz);
    thrust::device_vector<int> s_dev_cooColIndex(nnz);
    thrust::device_vector<int> s_dev_cooValIndex(nnz);
    thrust::device_vector<int> s_dev_csrRowPtr(nrows + 1);

    thrust::sequence(s_dev_cooRowIndex.begin(), s_dev_cooRowIndex.end());
    thrust::sequence(s_dev_cooColIndex.begin(), s_dev_cooColIndex.end());
    thrust::sequence(s_dev_cooValIndex.begin(), s_dev_cooValIndex.end());

    int* dev_cooRowIndex_ptr = thrust::raw_pointer_cast(&s_dev_cooRowIndex[0]);
    int* dev_cooColIndex_ptr = thrust::raw_pointer_cast(&s_dev_cooColIndex[0]);
    int* dev_cooValIndex_ptr = thrust::raw_pointer_cast(&s_dev_cooValIndex[0]);
    int* dev_csrRowPtr = thrust::raw_pointer_cast(&s_dev_csrRowPtr[0]);

    
    cusparseStatus_t status = cusparseCreate(&s_handle);

    // create and setup matrix descriptor
    status = cusparseCreateMatDescr(&s_stms_descr);
    if (status != CUSPARSE_STATUS_SUCCESS)
    {
        printf("Matrix descriptor initialization failed");
        assert(false);
    }
    cusparseSetMatType(s_stms_descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(s_stms_descr,CUSPARSE_INDEX_BASE_ZERO);

    // Transform STMs from COO to CSR format
    status = cusparseXcoo2csr(s_handle,
            dev_cooRowIndex_ptr,
            nnz,
            nrows,
            dev_csrRowPtr,
            CUSPARSE_INDEX_BASE_ZERO);
    assert(status == CUSPARSE_STATUS_SUCCESS);

    // for (int i = 0; i < nnz; i++)
    //     std::cout << s_dev_cooRowIndex[i] << " ";
    // std::cout << std::endl;
    // for (int i = 0; i < nnz; i++)
    //     std::cout << s_dev_cooColIndex[i] << " ";
    // std::cout << std::endl;
    // for (int i = 0; i < nnz; i++)
    //     std::cout << s_dev_cooValIndex[i] << " ";
    // std::cout << std::endl;
    // for (int i = 0; i < nrows + 1; i++)
    //     std::cout << s_dev_csrRowPtr[i] << " ";
    // std::cout << std::endl;
    return 0;
}