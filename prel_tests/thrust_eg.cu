#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/unique.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
// #include "read_velocity_file.h"
#include <iostream>

#define N 5
#define num_arrays 5
#define num_rzns 5

__global__ void fill_vector(int* a, int a_size){
    //a is raw pointer or device vector
    int tid = blockIdx.x;
    if (tid < a_size)
        a[tid] = 5 + (tid/2);
}


// __global__ void fill_arr_vectors( , ){


// }
void modify_H_vec(thrust::host_vector<int> &H_vec);
void modify_H_vec(thrust::host_vector<int> &H_vec){
    H_vec[0]=99;
}

void modify_D_vec(thrust::device_vector<int> &D_vec);
void modify_D_vec(thrust::device_vector<int> &D_vec){
    D_vec[0]=99;
}

// int main(){
//     // test copy array to vector
//     float a[10] = {1,2,1,2,3,1,2,3,1,2};
//     thrust::device_vector<float> H_vec(a, a+10);
//     for (int i = 0; i < 10; i++)
//         std::cout << a[i] << std::endl;

//     return 0;

// }




int main()
{
    // H_S2vec host vector to store S2
    thrust::host_vector<int> H_S2vec(N*num_rzns, 1);
    std::cout << "pre modify : H_S2vec[0] = " << H_S2vec[0] << std::endl;

    // test to see if pass thrust host vector by reference to a function works.
    modify_H_vec(H_S2vec);
    std::cout << "post modify : H_S2vec[0] = " << H_S2vec[0] << std::endl;


    thrust::device_vector<int> D_S2vec(N*num_rzns);
    modify_D_vec(D_S2vec);
    std::cout << "post modify : D_S2vec[0] = " << D_S2vec[0] << std::endl;


    int* D_s2arr = thrust::raw_pointer_cast(&D_S2vec[0]);

    
    // arr_H to copy restults from arr_D. These are array of vectors.
    thrust::host_vector<int> arr_H[num_arrays];
    thrust::host_vector<int> arr_D[num_arrays];

    for(int i = 0; i<num_arrays; i++){
        arr_D[i] = thrust::device_vector<int>(N);
    }

    // kernel call to fill D
    fill_vector<<<D_S2vec.size(),1>>>(D_s2arr, D_S2vec.size());

    // copy first N elements of filled D_S2vec to  arr_D[0]
    for(int i = 0; i < num_arrays; i ++)
        thrust::copy(D_S2vec.begin() + i*N, D_S2vec.begin() + (i+1)*N, arr_D[i].begin());

    // print arr_D
    for(int n = 0; n < num_arrays; n++){
        std::cout << "arr_D[" << n <<"]" << std::endl;
        for( int i = 0; i < N; i++)
            std::cout << arr_D[n][i] << std::endl;
    }

    for(int n = 0; n < num_arrays; n++)
        thrust::unique(arr_D[n].begin(), arr_D[n].end());

    // print arr_D
    for(int n = 0; n < num_arrays; n++){
        std::cout << "arr_D[" << n <<"]" << std::endl;
        for( int i = 0; i < N; i++)
            std::cout << arr_D[n][i] << std::endl;
        }
    


    // for(int i = 0; i< num_arrays; i++)
    //     thrust::copy(D_S2vec[], D_S2vec[], arr_D[i].begin())


    // thrust::sort(D.begin(), D.end());

    // // thrust::unique(D.begin(), D.end());
    // const int numberOfUniqueValues = thrust::unique(D.begin(),D.end()) - D.begin();
    // std::cout << "num of unique values = " << numberOfUniqueValues <<std::endl;

    // H and D are automatically deleted when the function returns
    return 0;
}