#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/generate.h>
#include <chrono>
using namespace std::chrono;

// ALSO CONTAINS EXPERIMETNS WITH SORT(THRUST stable_sort_by_key vs kernel of insertion sorts) + MANUAL REDUCE
//kernel of merge sort didnt work because it involved dynamically creating arrays , which seems not be possible in a kernel.
//among insertion and bubble sort, insertion sort was chosen because it was faster, when tested on cpu with in the 5k-10k range of array size

int num_actions = 8;
int ncells = 100*100;
int nrzns = 5000;
int arr_size = ncells * nrzns;
int n_print = 30;
int nblocks = ncells*num_actions;
int reps = 1000;

int my_mod_start = 0;

float my_mod(){
    int a = (my_mod_start)/nrzns;
    my_mod_start++;
    return (float)a;
  }

typedef thrust::device_vector<float>::iterator  dIter;


// count_kernel<<<ncells*num_actions,1>>>(ip_master_arr_ptr, nrzns, num_uq_s2_ptr)
__global__ void count_kernel(float* ip_master_arr_ptr, int nrzns, float* num_uq_s2_ptr) {
    //reduction by key and fills in the 3 op_arrays like coo rows.

    int tid = blockIdx.x;
    int nblocks = gridDim.x;  //ncells*num_actions
    float count = 0;
    float old_s2 = -1;
    float new_s2;
    int start_idx = tid*nrzns;

    if (tid < nblocks){
        
        for(int i = 0; i < nrzns; i++){
            new_s2 = ip_master_arr_ptr[start_idx + i];

            if ( new_s2 != old_s2){
                count++;
            }
            
            old_s2 = new_s2;
        }

        num_uq_s2_ptr[tid] = count;
    }
    return;
}

// reduce_kernel<<<nblocks,1>>>(ip_master_arr_ptr, nrzns, nnz_xa, op_s1_ptr, op_s2_ptr, op_cnt_ptr, num_uq_s2_ptr);
__global__ void reduce_kernel(float* ip_master_arr_ptr, int nrzns, int nnz_xa, float* op_s1_ptr, float* op_s2_ptr, float* op_cnt_ptr, float* num_uq_s2_ptr, float* prSum_num_uq_s2_ptr){
   
    int tid = blockIdx.x;
    int nblocks = gridDim.x;  //ncells*num_actions
    int start_idx = tid*nrzns; // to access tid'th threads 0-pos in ip_arr

    int n_uqs = num_uq_s2_ptr[tid]; //number of unique S2s for tid'th block
    int op_st_id = prSum_num_uq_s2_ptr[tid];   //sum of number of uniqeu S2s uptil tid'th block. to access tid'th thread's 0-pos in op_arr

    int ith_nuq = 0; //ranges from 0 to n_uqs , to index number between 0 and n_uqs

    float old_s2 = ip_master_arr_ptr[start_idx];
    float new_s2;
    float count = 0; //first if eval will lead to else condition and do  count++ 


    if (tid < nblocks){

        float s1 = tid; // TODO: change this to nbe a function of a arguments: sp_id and t
        for(int i = 0; i< n_uqs; i++)
            op_s1_ptr[op_st_id + i] = s1;

        for(int i = 0; i< nrzns; i++){
            new_s2 = ip_master_arr_ptr[start_idx + i];
            if (new_s2 != old_s2){                              // on encountering new value in the sorted array
                op_s2_ptr[op_st_id + ith_nuq] = old_s2;         // store old_s2 value in the [.. + ith] position
                op_cnt_ptr[op_st_id + ith_nuq] = count/nrzns;   // store prob value in the [.. + ith] position
                ith_nuq++;                                      // increment i
                count = 1;      //restart count on encounter new element
            }
            else
                count++;

            old_s2 = new_s2;

        }

        // to store information about the last of n_uqs S2s
        if (ith_nuq < n_uqs ){   //this condition should always be true because i assert ith_nuq == n_uqs - 1
            op_s2_ptr[op_st_id + ith_nuq] = old_s2;         // store old_s2 value in the [.. + ith] position
            op_cnt_ptr[op_st_id + ith_nuq] = count/nrzns;   // store prob value in the [.. + ith] position
            ith_nuq++;                                      // increment i
        }

   }
   return;
}


__device__ void insertionSort(float* arr, int n){  

    int i, key, j;  
    for (i = 1; i < n; i++) {  
        key = arr[i];  
        j = i - 1;  
  
        /* Move elements of arr[0..i-1], that are  
        greater than key, to one position ahead  
        of their current position */
        while (j >= 0 && arr[j] > key) 
        {  
            arr[j + 1] = arr[j];  
            j = j - 1;  
        }  
        arr[j + 1] = key;  
    }  
}  

__global__ void insertion_sort_master(float* ip_master_arr_ptr, int master_arr_size, int block_size){

    int tid = blockIdx.x;
    int nblocks = gridDim.x;  //ncells*num_actions
    int start_idx = tid*block_size; // to access tid'th threads 0-pos in ip_arr
                                    // block_size = nrzns

    if (tid < nblocks){
        insertionSort(ip_master_arr_ptr + start_idx, block_size);
        //sort array starting from star_idx and of size block_size
    }


}


int main(){
    // TEST: vectorised sort
    auto START = high_resolution_clock::now(); 

        // fill host array
        thrust::host_vector<float> H_S2_array(arr_size);
        for (int i = 0; i < arr_size; i++)
            H_S2_array[i] = (nrzns/reps) - (i%(nrzns/reps)); // to expect 100 reps of each integer after sort
        std::cout << std::endl;

        std::cout<< "S2_arrray" << std::endl;
        for (int i = 0; i < n_print; i++)
            std::cout<< H_S2_array[i] << " " ;
        std::cout << std::endl;


    auto init_Hvec = high_resolution_clock::now(); 

        // initialise array of device vecs
        thrust::device_vector<float> D_arr_of_S2_vecs[num_actions];
        for(int i = 0; i< num_actions; i++)
            D_arr_of_S2_vecs[i] = thrust::device_vector<float>(H_S2_array.begin(), H_S2_array.end());
 
    auto copy_H_to_D = high_resolution_clock::now(); 

        // maser vector for value::  this section takes 18.0716secs !!!
        thrust::host_vector<float> master_vals(arr_size*num_actions);
        // thrust::generate(master_vals.begin(), master_vals.end(), my_mod);
        for (int i = 0; i < arr_size*num_actions; i++)
            master_vals[i] = (int)(i/nrzns);

        // for(int i = 0; i < nrzns; i++)
        //     std::cout << master_vals[i] << ", ";
        
    
    auto generate  = high_resolution_clock::now(); 
        // check master_vals
       
        thrust::device_vector<float> D_master_vals(arr_size*num_actions);
        D_master_vals = master_vals;
        std::cout << "starting jugaad sort" << std::endl;

    auto start = high_resolution_clock::now(); 
        int master_arr_size = arr_size*num_actions;
        thrust::device_vector<float> master_S2_vector(master_arr_size);
        for(int i = 0; i< num_actions; i++)
            thrust::copy(D_arr_of_S2_vecs[i].begin(), D_arr_of_S2_vecs[i].end(), master_S2_vector.begin() + i*arr_size);

        std::cout << "master_S2_vector" << std::endl;
        for(int i = 0; i < n_print; i++)
            std::cout<< master_S2_vector[i] << " " ;
        std::cout << std::endl;

    auto mid = high_resolution_clock::now(); 

        float* ip_master_arr_ptr = thrust::raw_pointer_cast(&master_S2_vector[0]);
        thrust::stable_sort_by_key(master_S2_vector.begin(), master_S2_vector.end(), D_master_vals.begin());
        thrust::stable_sort_by_key(D_master_vals.begin(), D_master_vals.end(), master_S2_vector.begin());
        
        // insertion_sort_master<<<nblocks,1>>>(ip_master_arr_ptr, master_arr_size, nrzns);
        
        cudaDeviceSynchronize();

        std::cout << "post sort: master_S2_vector" << std::endl;
        for(int i = 0; i < n_print; i++)
            std::cout<< master_S2_vector[i] << " " ;
        std::cout << std::endl;

    auto end = high_resolution_clock::now(); 
    auto duration1 = duration_cast<microseconds>(end - start);
    std::cout << "copy + sort time = "<< duration1.count()/1e6 << std::endl;
    auto duration2 = duration_cast<microseconds>(end - mid);
    std::cout << "only sort time = "<< duration2.count()/1e6 << std::endl;

    thrust::device_vector<float> test_arr_n(100,-99);
    thrust::device_vector<float> test_arr_o(100,-99);

    thrust::device_vector<float> D_num_uq_s2(ncells*num_actions,0);
    thrust::device_vector<float> D_prSum_num_uq_s2(ncells*num_actions);




    float* num_uq_s2_ptr = thrust::raw_pointer_cast(&D_num_uq_s2[0]);
    float* prSum_num_uq_s2_ptr = thrust::raw_pointer_cast(&D_prSum_num_uq_s2[0]);

    float* test_arr_ptr_n = thrust::raw_pointer_cast(&test_arr_n[0]);
    float* test_arr_ptr_o = thrust::raw_pointer_cast(&test_arr_o[0]);

    auto red_start = high_resolution_clock::now(); 

        count_kernel<<<nblocks,1>>>(ip_master_arr_ptr, nrzns, num_uq_s2_ptr);

        int nnz_xa = (int) thrust::reduce(D_num_uq_s2.begin(), D_num_uq_s2.end(), (float) 0, thrust::plus<float>());
        thrust::device_vector<float> D_s1(nnz_xa);
        thrust::device_vector<float> D_s2(nnz_xa);
        thrust::device_vector<float> D_count(nnz_xa);
        float* op_s1_ptr = thrust::raw_pointer_cast(&D_s1[0]);
        float* op_s2_ptr = thrust::raw_pointer_cast(&D_s2[0]);
        float* op_cnt_ptr = thrust::raw_pointer_cast(&D_count[0]);

        thrust::exclusive_scan(D_num_uq_s2.begin(), D_num_uq_s2.end(), D_prSum_num_uq_s2.begin());

        reduce_kernel<<<nblocks,1>>>(ip_master_arr_ptr, nrzns, nnz_xa, op_s1_ptr, op_s2_ptr, op_cnt_ptr, num_uq_s2_ptr, prSum_num_uq_s2_ptr);

    auto red_end = high_resolution_clock::now(); 
    
    std::cout << "nnz_xa " << nnz_xa << std::endl;

    // check num_uq_s2_ptr
    std::cout << "num_uq " << std::endl;
    for(int i = 0; i < n_print; i++)
        std::cout << D_num_uq_s2[i] << std::endl;
    std::cout << std::endl;

    //check prefix sum
    std::cout << "post sum" << std::endl;
    for(int i = 0; i < 10; i++)
        std::cout << D_prSum_num_uq_s2[i] << std::endl;

    // check coo
    std::cout << "coo vals" << std::endl;
    for(int i = 0; i < n_print; i++)
        std::cout << D_s1[i] << " , " << D_s2[i] << " , " << D_count[i] << std::endl;
    std::cout << std::endl;







    auto red_duration = duration_cast<microseconds>(red_end - red_start);
    std::cout << "count+reduce kernels = "<< red_duration.count()/1e6 << std::endl;

    auto time_spent = duration_cast<microseconds>(init_Hvec - START);
    std::cout << "initialise H_vec = "<< time_spent.count()/1e6 << std::endl;
    
    time_spent = duration_cast<microseconds>(copy_H_to_D - init_Hvec);
    std::cout << "copy_H_to_D= "<< time_spent.count()/1e6 << std::endl;


    time_spent = duration_cast<microseconds>(generate - copy_H_to_D);
    std::cout << "generate= "<< time_spent.count()/1e6 << std::endl;

    time_spent = duration_cast<microseconds>(red_end - START);
    std::cout << "Total time= "<< time_spent.count()/1e6 << std::endl;


    // for (int i = 0; i < 10; i++){
    //     std::cout << "vec[" << i << "]" << std::endl; 
    //     for (int j = 0; j < 110; j++)
    //         std::cout<< D_red_S2[i][j] << " , " << D_red_counts[i][j] << std::endl;
    // }


    return 0;
}




// int main(){
//     // TEST: array of vectors do not form contiguous array elements

//     int num_actions = 8;
//     int ncells = 100*100;
//     int nrzns = 5000;
//     int arr_size = ncells * nrzns;

//     int n_vecs = 5;
//     int vec_size = 4;
//     thrust::device_vector<float> arr_of_vec[n_vecs];
//     for(int i = 0; i< n_vecs; i++)
//         arr_of_vec[i] = thrust::device_vector<float>(vec_size);
//     for(int i = 0; i< n_vecs; i++)
//         for(int j = 0; j< vec_size; j++)
//             arr_of_vec[i][j] = vec_size*i + j;

//     // std::cout << arr_of_vec[vec_size] << std::endl;
//     for(int i = 0; i< n_vecs; i++)
//         for(int j = 0; j< vec_size; j++)
//             std::cout << &arr_of_vec[i][j] << std::endl;
    
//     return 0;

// }





// int main(){
// // ----------------------------------------------------------
//     // TEST 3
//     //  sorting array of vectors in-array vs sorting vector chunks after copying data into chunks for each vector in array of vectors
//     // RESULTS: 
//     //  chunk based sorting is faster
//     // sorting vector in-array - 28.8 secs
//     // sorting vector chunks after copying data into chunks - 19.6 secs
// // ----------------------------------------------------------

//     int ncells = 100*100;
//     int nrzns = 5000;
//     int arr_size = ncells * nrzns;
//     int chunk_size = nrzns;
//     int n_print = 30;
//     int nchunks = arr_size/chunk_size;
//     int num_actions = 8;

//     // float S2_array[arr_size] = {1, 2, 3, 5, 2, 2, 4, 3, 4, 1 };

//     thrust::host_vector<float> H_S2_array(arr_size); //keys vector}

//     // fill host array
//     for (int i = 0; i < arr_size; i++)
//         H_S2_array[i] = i%(nrzns/10); // to expect 10 reps of each integer after sort
//     std::cout << std::endl;

//     for (int i = 0; i < n_print; i++)
//         std::cout<< H_S2_array[i] << std::endl;
//     std::cout << std::endl;


// // // ---------------------------------------------------------------------

// //     // array of S2_vecs
// //     thrust::device_vector<float> D_arr_of_S2_vecs1[num_actions];
// //     for(int i =0; i< num_actions; i++)
// //         D_arr_of_S2_vecs1[i] = thrust::device_vector<float>(H_S2_array.begin(), H_S2_array.end());

// //     auto start = high_resolution_clock::now(); 
    
// //         for (int i = 0; i< num_actions; i++)
// //             for (int j = 0; j< nchunks; j++)
// //                 thrust::sort(D_arr_of_S2_vecs1[i].begin() + j*chunk_size, D_arr_of_S2_vecs1[i].begin() + (j+1)*chunk_size);

// //     auto end = high_resolution_clock::now(); 
// //     auto duration = duration_cast<microseconds>(end - start);
// //     std::cout << "in-array sort time = "<< duration.count()/1e6 << std::endl;

// //     // RESULT : SORT TIME = 28.8 secs

// // // ---------------------------------------------------------------------

// // ---------------------------------------------------------------------  
//     // array of S2_vecs
//     thrust::device_vector<float> D_arr_of_S2_vecs2[num_actions];
//     for(int i =0; i< num_actions; i++)
//         D_arr_of_S2_vecs2[i] = thrust::device_vector<float>(H_S2_array.begin(), H_S2_array.end());


//     auto start = high_resolution_clock::now(); 

//         //make chunk vectors and copy data from main vector into chunks
//         thrust::device_vector<float> D_arr_of_chunk_vecs[num_actions][nchunks];
//         for (int i = 0; i < num_actions; i++)
//             for (int j = 0; j < nchunks; j++)
//                 D_arr_of_chunk_vecs[i][j] = thrust::device_vector<float> (chunk_size); 
        
//         for (int i = 0; i < num_actions; i++)
//             for (int j = 0; j < nchunks; j++)
//                 thrust::copy(D_arr_of_S2_vecs2[i].begin() + j*chunk_size, D_arr_of_S2_vecs2[i].begin() + (j+1)*chunk_size, 
//                                     D_arr_of_chunk_vecs[i][j].begin());
        
//         for (int i = 0; i < num_actions; i++)
//             for (int j = 0; j < nchunks; j++)
//                 thrust::sort(D_arr_of_chunk_vecs[i][j].begin(), D_arr_of_chunk_vecs[i][j].end());

//     auto end = high_resolution_clock::now(); 
//     auto duration = duration_cast<microseconds>(end - start);
//     std::cout << "copy-array sort time = "<< duration.count()/1e6 << std::endl;

//     // RESULT : SORT TIME = 19.6 secs
// // ---------------------------------------------------------------------  

//     return 0;
// }



// int main(){
//     // ----------------------------------------------------------
             // TEST 2
//         //  sorting vector in-array vs sorting vector chunks after copying data into chunks
//         // RESULTS: 
//         //  chunk based sorting is faster
//         // sorting vector in-array - 3.47465 secs
//         // sorting vector chunks after copying data into chunks - 2.3773 secs
//     // ----------------------------------------------------------
    
//         int ncells = 100*100;
//         int nrzns = 5000;
//         int arr_size = ncells * nrzns;
//         int chunk_size = nrzns;
//         int n_print = 30;
//         int nchunks = arr_size/chunk_size;
    
//         // float S2_array[arr_size] = {1, 2, 3, 5, 2, 2, 4, 3, 4, 1 };
    
//         thrust::host_vector<float> H_S2_array(arr_size); //keys vector}
    
//         // fill host array
//         for (int i = 0; i < arr_size; i++)
//             H_S2_array[i] = i%(nrzns/10); // to expect 10 reps of each integer after sort
//         std::cout << std::endl;
    
//         for (int i = 0; i < n_print; i++)
//             std::cout<< H_S2_array[i] << std::endl;
//         std::cout << std::endl;
    
//         thrust::device_vector<float> D_S2_array_1(arr_size);
//         thrust::device_vector<float> D_S2_array_2(arr_size);
    
//         D_S2_array_1 = H_S2_array;
//         D_S2_array_2 = H_S2_array;
    
//         //  Sort 1 dec_vector in-array
//         auto start = high_resolution_clock::now(); 
        
//             for (int i = 0; i< nchunks; i++)
//                 thrust::sort(D_S2_array_1.begin() + i*chunk_size, D_S2_array_1.begin() + (i+1)*chunk_size);
    
//         auto end = high_resolution_clock::now(); 
//         auto duration = duration_cast<microseconds>(end - start);
//         std::cout << "in-array sort time = "<< duration.count()/1e6 << std::endl;
    
//         //check sorted resulsts - OK
//         std::cout << "sorted array "<< std::endl;
//         for (int i = 0; i < n_print; i++)
//             std::cout<< D_S2_array_1[i] << std::endl;
//         std::cout << std::endl ;
    
    
//         start = high_resolution_clock::now(); 
    
//             //make chunk vectors and copy data from main vector into chunks
//             thrust::device_vector<float> D_arr_of_chunk_vecs[nchunks];
//             for (int i = 0; i < nchunks; i++)
//                 D_arr_of_chunk_vecs[i] = thrust::device_vector<float> (chunk_size);
    
//             for (int i = 0; i < nchunks; i++)
//                 thrust::copy(D_S2_array_2.begin() + i*chunk_size, D_S2_array_2.begin() + (i+1)*chunk_size, 
//                                     D_arr_of_chunk_vecs[i].begin());
            
//             for (int i = 0; i < nchunks; i++)
//                 thrust::sort(D_arr_of_chunk_vecs[i].begin(), D_arr_of_chunk_vecs[i].end());
    
//         end = high_resolution_clock::now(); 
//         duration = duration_cast<microseconds>(end - start);
//         std::cout << "copy-array sort time = "<< duration.count()/1e6 << std::endl;
    
//         //check sorted resulsts - OK
//         std::cout << "sorted array " << std::endl;
//         for (int k = 0; k < 3; k++){
//             std::cout << "------chunk " << k << std::endl;
//             for (int i = 0; i < n_print; i++)
//                 std::cout<< D_arr_of_chunk_vecs[k][i] << std::endl;
//         }
//         std::cout << std::endl;
    
    
//         return 0;
//     }







// int main(){
// // ----------------------------------------------------------
        // TEST 1
//     //  sorting in chunks over a single vector works !!
//     // SOLUTION: 1 2 2 3 5 1 2 3 4 4
// // ----------------------------------------------------------

//     int arr_size = 10;
//     int chunk_size = 5;
//     float S2_array[arr_size] = {1, 2, 3, 5, 2, 2, 4, 3, 4, 1 };

//     thrust::device_vector<float> D_S2_array(S2_array, S2_array + arr_size); //keys vector}

//     int nchunks = arr_size/chunk_size;
//     for (int i = 0; i< nchunks; i++)
//         thrust::sort(D_S2_array.begin() + i*chunk_size, D_S2_array.begin() + (i+1)*chunk_size);

//     for (int i = 0; i < arr_size; i++)
//         std::cout<< D_S2_array[i] << std::endl;
//     std::cout << std::endl;

//     return 0;

// }