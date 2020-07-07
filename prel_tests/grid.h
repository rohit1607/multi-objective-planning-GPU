#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

void define_xs_or_ys(float* xs, float dx, float x0, int gsize);

void define_xs_or_ys(float* xs, float dx, float x0, int gsize){

    for(int i = 0; i < gsize;  i++)
        xs[i] = x0 + i*dx;
}