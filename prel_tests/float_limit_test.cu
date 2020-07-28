#include<iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

int main(){
    int size = (int)1.6e8;
    thrust::host_vector<int32_t> test(size);

    for (int i = 0; i < size; i++)
        test[i] = i;

    for (int i = size - 5000 ; i < size; i++) 
        std::cout << test[i] << " " ;

    return 0;
}