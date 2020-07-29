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



// template<typename T>
// void f(T s)
// {
//     std::cout << s << '\n';
// }
 
// int main()
// {
//     f<double>(1); // instantiates and calls f<double>(double)
//     f<>('a'); // instantiates and calls f<char>(char)
//     f(7); // instantiates and calls f<int>(int)
//     void (*ptr)(std::string) = f; // instantiates f<string>(string)
// }




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



int main(){

    std::string data_path = "data/nT_60/";
    std::string all_u_fname = data_path + "all_u_mat.npy";
    int all_u_n_elms;
    cnpy::NpyArray all_u_cnpy = read_velocity_field_data<float>(all_u_fname, &all_u_n_elms);

}
