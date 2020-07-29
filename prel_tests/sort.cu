#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/generate.h>
#include <chrono>
using namespace std::chrono;


int num_actions = 8;
int ncells = 100*100;
int nrzns = 5000;
int arr_size = ncells * nrzns;
int n_print = 30;

int my_mod_start = 0;

float my_mod(){
    int a = (my_mod_start)/nrzns;
    my_mod_start++;
    return (float)a;
  }

typedef thrust::device_vector<float>::iterator  dIter;

int main(){
    // TEST: vectorised sort
    auto START = high_resolution_clock::now(); 

        // fill host array
        thrust::host_vector<float> H_S2_array(arr_size);
        for (int i = 0; i < arr_size; i++)
            H_S2_array[i] = i%(nrzns/100); // to expect 100 reps of each integer after sort
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

        thrust::device_vector<float> master_S2_vector(arr_size*num_actions);
        for(int i = 0; i< num_actions; i++)
            thrust::copy(D_arr_of_S2_vecs[i].begin(), D_arr_of_S2_vecs[i].end(), master_S2_vector.begin() + i*arr_size);

        // for(int i = 0; i < arr_size*3; i++)
        //     std::cout<< master_S2_vector[i] << ", " ;
        // std::cout << std::endl;


    auto mid = high_resolution_clock::now(); 

        thrust::stable_sort_by_key(master_S2_vector.begin(), master_S2_vector.end(), D_master_vals.begin());
        thrust::stable_sort_by_key(D_master_vals.begin(), D_master_vals.end(), master_S2_vector.begin());
        cudaDeviceSynchronize();

        // for(int i = 0; i < arr_size*3; i++)
        //     std::cout<< master_S2_vector[i] << ", " ;
        // std::cout << std::endl;

    auto end = high_resolution_clock::now(); 
    auto duration1 = duration_cast<microseconds>(end - start);
    std::cout << "copy + sort time = "<< duration1.count()/1e6 << std::endl;
    auto duration2 = duration_cast<microseconds>(end - mid);
    std::cout << "only sort time = "<< duration2.count()/1e6 << std::endl;



    thrust::device_vector<float> D_ones(nrzns, 1);
    int num_vecs = arr_size * num_actions / nrzns ;
    thrust::device_vector<float> D_red_S2[num_vecs];
    thrust::device_vector<float> D_red_counts[num_vecs];
    for (int i = 0; i < num_vecs; i++){
        D_red_S2[i] = thrust::device_vector<float>(nrzns);
        D_red_counts[i] = thrust::device_vector<float>(nrzns);
    }
    thrust::device_vector<float> D_redS2_size(num_vecs);
    thrust::pair<dIter, dIter> new_end;

    auto red_start = high_resolution_clock::now(); 
        // This section takes 3 seconds
        for (int i = 0; i < num_vecs; i++){
            new_end = thrust::reduce_by_key(master_S2_vector.begin() + (i*nrzns), master_S2_vector.begin() + ((i+1)*nrzns), 
                                    D_ones.begin(), D_red_S2[i].begin(), D_red_counts[i].begin());
            
            // D_redS2_size[i] = new_end.first - &D_red_S2[i][0];
            // std::cout << D_redS2_size[i] << std::endl;
        }
    auto red_end = high_resolution_clock::now(); 
    auto red_duration = duration_cast<microseconds>(red_end - red_start);
    std::cout << "reduce_by_key = "<< red_duration.count()/1e6 << std::endl;

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