#include"cnpy.h"
#include<cstdlib>
#include<iostream>
#include <string>
// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>


// float* read_velocity_field_data(float* vel_data, char* file_path_name);

// float* read_velocity_field_data(float* vel_data, char* file_path_name){

cnpy::NpyArray read_velocity_field_data( std::string file_path_name, int* n_elements);

cnpy::NpyArray read_velocity_field_data( std::string file_path_name, int* n_elements){
    // reads numpy file from input and 
    // returns cnpy::NpyArray stucture 
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


int main()
{
    // float* data;
    // data = read_velocity_field_data("fl32_24.npy");
    std::string fname = "data/fl32_24.npy";
    int num_elements = -1;
    cnpy::NpyArray arr = read_velocity_field_data(fname, &num_elements);
    std::cout << "n_elements = " << num_elements << std::endl;
    float* vel_data = arr.data<float>();
    std::cout << std::endl << "In main: First 10 elements of loaded array are: " << std::endl;
    for (int i = 0; i < 24; i++)
        std::cout << vel_data[i] << "  " ;
    
    return 0;

    // // TODO: Low priority - directly load npy data to hvec
    // cnpy::NpyArray arr = cnpy::npy_load("f32_24.npy");
    // std::cout << "shape "<< arr.shape[0] << " , " << arr.shape[1] << " , " << arr.shape[2] << std::endl;
    // std::cout << "arr.word_size = " << arr.word_size << std::endl;
    // float* loaded_array =   arr.data<float>();
    // std::cout << "loaded_data"<< std::endl;
    // for( int i = 0; i< 24; i++)
    //     std::cout << loaded_array[i] << std::endl;

    // thrust::host_vector<float> H_vec(24, -1);
    // // float* H_arr = thrust::raw_pointer_cast(&H_vec[0]);
    // // thrust::host_vector<float> *H_vec = arr.data<thrust::host_vector<float>>();
    // // std::vector<float>* H_vec = arr.data<std::vector<float>>();

    // for( int i = 0; i< 24; i++)
    //    H_vec[i] = loaded_array[i];

    // std::cout << std::endl << "post fill H_vec"<< std::endl;
    // for( int i = 0; i< 24; i++)
    //     std::cout << H_vec[i] << std::endl;

    
    

    // //set random seed so that result is reproducible (for testing)
    // srand(0);
    // //create random data
    // std::vector<std::complex<double>> data(Nx*Ny*Nz);
    // for(int i = 0;i < Nx*Ny*Nz;i++) data[i] = std::complex<double>(rand(),rand());

    // //save it to file
    // cnpy::npy_save("arr1.npy",&data[0],{Nz,Ny,Nx},"w");

    // //load it into a new array
    // cnpy::NpyArray arr = cnpy::npy_load("arr1.npy");
    // std::complex<double>* loaded_data = arr.data<std::complex<double>>();
    
    // //make sure the loaded data matches the saved data
    // assert(arr.word_size == sizeof(std::complex<double>));
    // assert(arr.shape.size() == 3 && arr.shape[0] == Nz && arr.shape[1] == Ny && arr.shape[2] == Nx);
    // for(int i = 0; i < Nx*Ny*Nz;i++) assert(data[i] == loaded_data[i]);

    // //append the same data to file
    // //npy array on file now has shape (Nz+Nz,Ny,Nx)
    // cnpy::npy_save("arr1.npy",&data[0],{Nz,Ny,Nx},"a");

    // //now write to an npz file
    // //non-array variables are treated as 1D arrays with 1 element
    // double myVar1 = 1.2;
    // char myVar2 = 'a';
    // cnpy::npz_save("out.npz","myVar1",&myVar1,{1},"w"); //"w" overwrites any existing file
    // cnpy::npz_save("out.npz","myVar2",&myVar2,{1},"a"); //"a" appends to the file we created above
    // cnpy::npz_save("out.npz","arr1",&data[0],{Nz,Ny,Nx},"a"); //"a" appends to the file we created above

    // //load a single var from the npz file
    // cnpy::NpyArray arr2 = cnpy::npz_load("out.npz","arr1");

    // //load the entire npz file
    // cnpy::npz_t my_npz = cnpy::npz_load("out.npz");
    
    // //check that the loaded myVar1 matches myVar1
    // cnpy::NpyArray arr_mv1 = my_npz["myVar1"];
    // double* mv1 = arr_mv1.data<double>();
    // assert(arr_mv1.shape.size() == 1 && arr_mv1.shape[0] == 1);
    // assert(mv1[0] == myVar1);
}

