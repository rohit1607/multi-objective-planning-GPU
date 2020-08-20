run: build_model
	./bin/build_model

build_model: build_model.o get_funcs.o extract_field.o move_and_rewards.o utils.o
	nvcc --gpu-architecture=sm_50 build_model.o get_funcs.o extract_field.o move_and_rewards.o utils.o -L/usr/local/ -lcnpy -lz --std=c++11 --output-file bin/build_model

build_model.o: build_model.cu
	nvcc --gpu-architecture=sm_50 --device-c build_model.cu

get_funcs.o: get_funcs.cu
	nvcc --gpu-architecture=sm_50 --device-c get_funcs.cu

extract_field.o: extract_field.cu
	nvcc --gpu-architecture=sm_50 --device-c extract_field.cu

move_and_rewards.o: move_and_rewards.cu
	nvcc --gpu-architecture=sm_50 --device-c move_and_rewards.cu

utils.o: utils.h
	nvcc --gpu-architecture=sm_50 --device-c utils.cu

clean:
	rm -f build_model.o get_funcs.o extract_field.o move_and_rewards.o
	rm -f bin/build_model