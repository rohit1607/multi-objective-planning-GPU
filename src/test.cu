#include <thrust/host_vector.h>
#include "utils.h"

int main(){

    int num_ac_speeds = 3;
    int num_ac_angles = 4;
    int num_actions = num_ac_speeds*num_ac_angles;
    float F = 2;

    // thrust::host_vector<float> H_actions[num_actions];
    // for (int i=0; i<num_actions; i++)
    //     H_actions[i] = thrust::host_vector<float>(2);
    // float H_actions[num_actions][2];
    float** H_actions = new float*[num_actions];
    for(int i=0; i<num_actions; i++)
        H_actions[i] = new float[2];
    
    for(int i=0; i<num_actions; i++){
        H_actions[i][0] = i;
        H_actions[i][1] = i+1;
    }

    for(int i=0; i<num_actions; i++){
        std::cout << H_actions[i][0] << ", " << H_actions[i][1] << "\n";
    }
    populate_actions(H_actions, num_ac_speeds, num_ac_angles, F);
    for(int i=0; i<num_actions; i++){
        std::cout << H_actions[i][0] << ", " << H_actions[i][1] << "\n";
    }
    return 0;
}
