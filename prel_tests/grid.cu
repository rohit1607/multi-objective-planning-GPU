#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

void define_xs_or_ys(float* xs, float dx, float x0, int gsize);

void define_xs_or_ys(float* xs, float dx, float x0, int gsize){

    for(int i = 0; i < gsize;  i++)
        xs[i] = x0 + i*dx;
}

int main(){
    int gsize = 10;
    float dx = 1;
    float x0 = 0.5;
    thrust::host_vector<float> H_xs(gsize, -1);
    thrust::host_vector<float> H_ys(gsize, -1);
    float* xs = thrust::raw_pointer_cast(&H_xs[0]);
    float* ys = thrust::raw_pointer_cast(&H_ys[0]);

    define_xs_or_ys(xs, dx, x0, gsize);

    for(int i = 0; i < gsize;  i++)
        std::cout << xs[i] << std::endl;

}