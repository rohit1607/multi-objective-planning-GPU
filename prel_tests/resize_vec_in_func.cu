#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>


// TEST to see if i can resize vector in a function

int new_size = 10;
int dev_init = 33;
void resize_vec(thrust::host_vector<int> &vec){

    vec.resize(new_size);
    // for( int i = 0; i< new_size; i++)
    //     vec[i] = i;
    thrust::device_vector<int> d_vec(new_size, dev_init);
    int* d_vec_ptr = thrust::raw_pointer_cast(&d_vec[0]);
    int* vec_ptr = thrust::raw_pointer_cast(&vec[0]);

    cudaMemcpy(vec_ptr, d_vec_ptr, (size_t)(new_size*sizeof(int)), cudaMemcpyDeviceToHost);

}

int main(){

    thrust::host_vector<int> test_vec(0);
    std::cout << "Old size = " << test_vec.size() << std::endl;

    resize_vec(test_vec);
    std::cout << "new size = " << test_vec.size() << std::endl;

    for( int i = 0; i< new_size; i++)
        std::cout << test_vec[i] << std::endl;

    return 0;
}