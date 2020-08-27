#include "utils.h"
#include <iostream>


void save_master_Coos_to_file(std::string op_FnamePfx, int num_actions,
    thrust::host_vector<long long int> &H_master_cooS1, 
    thrust::host_vector<long long int> &H_master_cooS2, 
    thrust::host_vector<float> &H_master_cooVal,
    thrust::host_vector<float> &H_master_R,
    thrust::host_vector<long long int>* H_Aarr_of_cooS1,
    thrust::host_vector<long long int>* H_Aarr_of_cooS2,
    thrust::host_vector<float>* H_Aarr_of_cooProb,
    thrust::host_vector<float>* H_Aarr_of_Rs,
    thrust::host_vector<float> &prob_params,
    long long int* DP_relv_params,
    unsigned long int num_DP_params){
    //  Convertes floats to int32 for COO row and col idxs
    //  copies from each action vector to a master vector
    //  master_coo vectors is concatation first across time, then across action
    //  ALSO, MODIFIES S1(t,i,j) to S1(t,i,j,a)

    unsigned long long int master_nnz = H_master_cooS1.size();
    unsigned long long int prob_params_size = prob_params.size();
    int m_idx = 0;
    int n_states = DP_relv_params[0];

    std::cout << "in save \n" ;

    for(int i = 0; i < num_actions; i++){
        for(int j = 0; j< H_Aarr_of_cooS1[i].size(); j++){
            // TODO: modify to include actions
            H_master_cooS1[m_idx] = H_Aarr_of_cooS1[i][j] + i*n_states;
            m_idx++;
        }
    }

    std::cout << "in save \n" ;
    m_idx = 0;
    for(int i = 0; i < num_actions; i++){
        for(int j = 0; j< H_Aarr_of_cooS2[i].size(); j++){
            H_master_cooS2[m_idx] = H_Aarr_of_cooS2[i][j];
            m_idx++;
        }
    }

    std::cout << "in save \n" ;
    m_idx = 0;
    for(int i = 0; i < num_actions; i++){
        for(int j = 0; j< H_Aarr_of_cooProb[i].size(); j++){
            H_master_cooVal[m_idx] = H_Aarr_of_cooProb[i][j];
            m_idx++;
        }
    }

    std::cout << "in save \n" ;
    m_idx = 0;
    for(int i = 0; i < num_actions; i++){
        for(int j = 0; j< H_Aarr_of_Rs[i].size(); j++){
            H_master_R[m_idx] = H_Aarr_of_Rs[i][j];
            m_idx++;
        }
    }

    
    std::cout << "check num_DP_params = " << num_DP_params << std::endl;
    std::cout << "op_FnamePfx= " <<  op_FnamePfx << "\n";
    
    cnpy::npy_save(op_FnamePfx + "master_cooS1.npy", &H_master_cooS1[0], {master_nnz,1},"w");
    cnpy::npy_save(op_FnamePfx + "master_cooS2.npy", &H_master_cooS2[0], {master_nnz,1},"w");
    cnpy::npy_save(op_FnamePfx + "master_cooVal.npy", &H_master_cooVal[0], {master_nnz,1},"w");
    cnpy::npy_save(op_FnamePfx + "master_R.npy", &H_master_R[0], {H_master_R.size(),1},"w");
    cnpy::npy_save(op_FnamePfx + "DP_relv_params.npy", &DP_relv_params[0], {num_DP_params,1},"w");
    cnpy::npy_save(op_FnamePfx + "prob_params.npy", &prob_params[0], {prob_params_size,1},"w");

}



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

    float* vel_data = arr.data<float>();
    // print check first 10 elements
    std::cout << "First 10 elements of loaded array are: " << std::endl;
    for (int i = 0; i < 10; i++)
         std::cout << vel_data[i] << "  " ;
    
    std::cout << std::endl << std::endl;

    return arr;

}



// template<typename dType>
void print_device_vector( thrust::device_vector<long long int> &array, int start_id, int end_id, std::string array_name, std::string end, int method){
    std::cout << array_name << "  from id " << start_id << "  to  " << end_id << std::endl;
    if (method == 1){
        float temp = -10000000;
        for(int i = start_id; i < end_id; i++){
            if (array[i] != temp){
                std::cout << i << "\n";
                std::cout << array[i] << " " << end;
                std::cout << "\n";
                temp = array[i];
            }
        }
    }

    else if (method == 0){
        for(int i = start_id; i < end_id; i++)
            std::cout << array[i] << " " << end;
    }

    else
        std::cout << "Invalid input for argument: method";


    std::cout << std::endl;
}


void make_dir(std::string dir_name){
    int mkdir_status;
    std::string comm_mkdir = "mkdir ";
    std::string str = comm_mkdir + dir_name;
    const char * full_command = str.c_str();
    mkdir_status = system(full_command);
    std::cout << "mkdir_status = " << mkdir_status << std::endl;
}



void define_xs_or_ys(float* xs, float dx, float x0, int gsize){

    for(int i = 0; i < gsize;  i++)
        xs[i] = x0 + i*dx;
}



void populate_ac_angles(float* ac_angles, int num_ac_angles){
    //fills array with equally spaced angles in radians
    for (int i = 0; i < num_ac_angles; i++)
        ac_angles[i] = i*(2*M_PI)/num_ac_angles;
    return;
}



void populate_ac_speeds(float* ac_speeds, int num_ac_speeds, float Fmax){
    //fills array with ac_speeds
    std::cout << "infunc CHeck- num_ac_speeds = " << num_ac_speeds << "\n";
    int delF = 0;
    if (num_ac_speeds == 1)
        ac_speeds[0] = Fmax;
    else if (num_ac_speeds > 1){
        delF = Fmax/(num_ac_speeds-1);
        for(int i = 0; i<num_ac_speeds; i++)
            ac_speeds[i] = i*delF;
    }
    else
        std::cout << "Invalid num_ac_speeds\n";
    
    return;
}


void populate_actions(float **H_actions, int num_ac_speeds, int num_ac_angles, float Fmax){
    // populates 2d vector with possible actions
    float* ac_angles = new float[num_ac_angles];
    populate_ac_angles(ac_angles, num_ac_angles);

    float* ac_speeds = new float[num_ac_speeds];
    populate_ac_speeds(ac_speeds, num_ac_speeds, Fmax);

    int idx;
    for (int i=0; i<num_ac_speeds; i++){
        for(int j=0; j<num_ac_angles; j++){
            idx = j + num_ac_angles*i;
            H_actions[idx][0] = ac_speeds[i];
            H_actions[idx][1] = ac_angles[j];
        }
    }

    return;
}


