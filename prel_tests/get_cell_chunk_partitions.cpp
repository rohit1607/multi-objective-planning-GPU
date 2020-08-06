#include<iostream>
using namespace std;

long long int GPUmem = 8*1000*1000*1000LL; // using 1000 instead of 1024 
void get_cell_chunk_partition(int gsize, int ncells, int nrzns, int num_actions,
        int nmodes, int nt, int thrust_fraction, int avg_nnz_per_row,
        int* nchunks, int* chunk_size, int* last_chunk_size){
    // reads dimensions of input data related to the problem and returns sizes
    // and number of chunks into which ncells (spatial grid) is divided.
    // So as to be able to fit all necesarry data structures in GPU memory

    long long int master_arr_size_term = 8*nrzns*num_actions;
    long long int vdata_size_term = 8*nt*(nmodes+1);
    long long int coo_term = 3*avg_nnz_per_row*num_actions;
    int k = thrust_fraction;
    long long int denom = ((1+k)*master_arr_size_term) + vdata_size_term;

    cout << "master_arr_size_term= " << master_arr_size_term << "\n" 
        << "vdata_size_term = " <<vdata_size_term << "\n"
        << "denom = " << denom << "\n" ;

    int local_chunk_size = (int) (GPUmem/denom);
    if (local_chunk_size < ncells){
        *chunk_size = local_chunk_size;
        *nchunks = (ncells/local_chunk_size) + 1;
        *last_chunk_size = ncells - ( (local_chunk_size)*(*nchunks - 1) );
    }

    return;
}



int main(){

int gsize = 1e2;
int ncells = gsize*gsize;
int nrzns = 5000;
int num_actions = 16;
int nmodes  = 1;
int nt = 100;
int nchunks = 1; 
int chunk_size = ncells;
int last_chunk_size = 0;
int thrust_fraction = 1; //expected  memory use for thrust method calls.
int avg_nnz_per_row = 10; // avg num  of nnz per row

get_cell_chunk_partition(gsize, ncells, nrzns, num_actions,
    nmodes, nt, thrust_fraction, avg_nnz_per_row, &nchunks, &chunk_size, &last_chunk_size);

cout << "nchunks = " << nchunks << "\n" 
        << "chunk_size = " << chunk_size << "\n" 
        << "last_chunk_size = " << last_chunk_size << "\n";

return 0;
}