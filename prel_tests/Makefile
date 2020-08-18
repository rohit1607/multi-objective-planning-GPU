
build_model_test: build_model_test.o get_funcs.o
	nvcc --gpu-architecture=sm_50 build_model_split_test.o get_funcs.o -L/usr/local/ -lcnpy -lz --std=c++11 --output-file bin/build_model_test

build_model_test.o: build_model_split_test.cu
	nvcc --gpu-architecture=sm_50 --device-c build_model_split_test.cu

get_funcs.o: get_funcs.cu
	nvcc --gpu-architecture=sm_50 --device-c get_funcs.cu

clean:
	rm -f build_model_test.o get_funcs.o
	rm -f bin/build_model_test