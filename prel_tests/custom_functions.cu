#include<iostream>
#include<cmath>
#include<thrust/host_vector.h>
using namespace std;

void populate_ac_angles(float* ac_angles, int num_actions);
void populate_ac_angles(float* ac_angles, int num_actions){
    //fills array with equally spaced angles in radians
    for (int i = 0; i < num_actions; i++)
        ac_angles[i] = i*(2*M_PI)/num_actions;

}

int main(){

    int num_actions = 8;
    thrust::host_vector<float> H_ac_angles(num_actions);
    float* ac_angles = thrust::raw_pointer_cast(&H_ac_angles[0]);

    //TODO: write function populate_ac_angles
    populate_ac_angles(ac_angles, num_actions);

    for (int i = 0; i < num_actions; i++)
        cout << H_ac_angles[i] << endl;
}