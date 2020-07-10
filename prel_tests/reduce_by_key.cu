#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
// #include "read_velocity_file.h"
#include <iostream>

typedef thrust::device_vector<float>::iterator  dIter;

int main(){

    int num_actions = 8;
    int ncells = 25;
    int nrzns = 10;
    int S2_vec_size = ncells*nrzns;

    // intialise S2_vectors for each action
    thrust::device_vector<float> D_array_of_S2_vectors[num_actions];
    for (int i =0 ; i < num_actions ; i++)
        D_array_of_S2_vectors[i] = thrust::device_vector<float>(S2_vec_size, -1);

    // fill vecs for all actions with 4 consec  repeating numbers  111122223333 ....
    for (int i = 0; i< num_actions; i++)
        for (int j = 0; j< S2_vec_size; j++)
            D_array_of_S2_vectors[i][j] = 1000*i + j/4;

    // value vector of 1s to act as values for reduction - We just need one. saves memory.
    // IMP: however, if the reduce_by-keys are asynchorous, then may be have multiple such vectors.
    // TODO: check above fact.
    thrust::device_vector<float> D_ones(nrzns, 1);

    // Question: sort array in blocks (is it even possible) vs make copy of blocks vs sort array in blocks






    return 0;
}



























int main(){

    // ______ reduce_by_key() on a single device vector_____ SUCCESS_____ 

    int arr_size = 10;
    float S2_array[arr_size] = {1, 2, 3, 5, 2, 2, 4, 3, 4, 1 };
    for (int i = 0; i < arr_size; i ++)
        std::cout<< S2_array[i] << std::endl;
    std::cout << std::endl;

    thrust::device_vector<float> D_S2_array(S2_array, S2_array + arr_size); //keys vector

    // for (int i = 0; i < arr_size; i ++)
    //     std::cout<< D_S2_array[i] << std::endl;
    // std::cout << std::endl;

    thrust::device_vector<float> D_ones(arr_size, 1);     //vals vector
    thrust::device_vector<float> D_red_keys(arr_size);
    thrust::device_vector<float> D_red_vals(arr_size);

    thrust::sort(D_S2_array.begin(), D_S2_array.end());

    std::cout << "After sort" << std::endl;
    // for (int i = 0; i < arr_size; i ++)
    //     std::cout<< D_S2_array[i] << std::endl;
    // std::cout << std::endl;

    thrust::pair<dIter, dIter> new_end;
    new_end = thrust::reduce_by_key(D_S2_array.begin(), D_S2_array.end(), D_ones.begin(), D_red_keys.begin(), D_red_vals.begin() );

    // int num_elements_f = new_end.first() - D_red_keys.begin();
    // int num_elements_s = new_end.second() - D_red_vals.begin();
    // std::cout << "num_elements_f, num_elements_s = " << new_end.first() - D_red_keys.begin()<< " , " <<new_end.second() - D_red_vals.begin()<< std::endl;

    for (int i = 0; i < 5; i ++)
        std::cout<< D_red_keys[i] << "  , " << D_red_vals [i] << std::endl;
    std::cout << std::endl; 


}