#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <chrono>
using namespace std::chrono;

// comparison between insert vs copy vs resize


int main(){
    // c1. larger vec to smaller vec -> in
    int n = 1000000;
    int reps = 8;
    int init_size = n*reps/2;
    thrust::host_vector<float> tmaster(0);
    thrust::device_vector<float> temp(n, 1);
    auto START = high_resolution_clock::now(); 

        for(int i = 0; i < reps; i++){
            tmaster.insert(tmaster.end(), temp.begin(), temp.end());
        }
    auto End = high_resolution_clock::now(); 
    // for( int i = 0; i < tmaster.size(); i++)
    //     std::cout << tmaster[i] << " ";

    auto red_duration = duration_cast<microseconds>(End - START);
    std::cout << "insert time = "<< red_duration.count()/1e6 << std::endl;
    
    std::cout << std::endl;
    std::cout << "tmaster.size()= " << tmaster.size() << std::endl;

    return 0;

}

// int main(){
//     // c1. larger vec to smaller vec -> in
//     int n = 1000000;
//     int reps = 8;
//     int init_size = n*reps/2;
//     thrust::host_vector<float> tmaster(2*n*reps);
//     thrust::device_vector<float> temp(n, 1);
//     auto START = high_resolution_clock::now(); 

//         for(int i = 0; i < reps; i++){
//             thrust::copy(temp.begin(), temp.end(), tmaster.begin() + i*n);
//         }
//         tmaster.resize(n*reps);
//     auto End = high_resolution_clock::now(); 
//     // for( int i = 0; i < tmaster.size(); i++)
//     //     std::cout << tmaster[i] << " ";

//     auto red_duration = duration_cast<microseconds>(End - START);
//     std::cout << "copy + resize time = "<< red_duration.count()/1e6 << std::endl;
    
//     std::cout << std::endl;
//     std::cout << "tmaster.size()= " << tmaster.size() << std::endl;

//     return 0;

// }


