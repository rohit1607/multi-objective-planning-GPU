
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

int n = 5;
void fill_arr_of_vector(thrust::host_vector<float>* arr_of_vec, int arr_size, int i){

    thrust::host_vector<float> temp(n,i);
    for( int i = 0; i < arr_size; i++){
        arr_of_vec[i].insert(arr_of_vec[i].end(), temp.begin(), temp.end());
    }
}

int main(){
    int num_actions = 8;
    int nt = 4;
    thrust::host_vector<float> arr_of_vec[num_actions];
    for(int i=0; i<num_actions; i++)
        arr_of_vec[i] = thrust::host_vector<float> (0);
    
    for(int i=0; i<nt; i++)
        fill_arr_of_vector(arr_of_vec, num_actions, i);

    for(int i=0; i<num_actions; i++){
        for( int j =0; j<arr_of_vec[i].size(); j++)
            std::cout << arr_of_vec[i][j] << " ";
        std::cout << std::endl;
    }


    return 0;
}