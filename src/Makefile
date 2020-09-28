all: run_build_model_and_run_solver
	python ../Post_processing/plot_e2eGPU_policy.py

plot_only:
	python ../Post_processing/plot_e2eGPU_policy.py

run_build_model_and_run_solver: run_build_model_and_convert_coo run_spvi_only
	echo "run_build_model_and_run_solver"

run_spvi_only: spvi
	CUDA_VISIBLE_DEVICES=0 ./bin/spvi_2
	
spvi: spvi_2.cu
	nvcc -o bin/spvi_2 spvi_2.cu -L/usr/local/ -lcnpy -lz -L/usr/local/cuda/lib64 -lcusparse -lcudart --std=c++11

run_build_model_and_convert_coo: build_model
	CUDA_VISIBLE_DEVICES=0 ./bin/build_model
	python convert_model_datatype.py
	cp temp_modelOp_dirName.txt ../Post_processing/temp_modelOp_dirName.txt

build_model: 
	nvcc -o bin/build_model build_model.cu -L/usr/local/ -lcnpy -lz --std=c++11

clean:
	rm -f build_model.o get_funcs.o extract_field.o move_and_rewards.o utils.o
	rm -f bin/build_model bin/spvi_2
	rm -f temp_modelOp_dirName.txt

clean_spvi: 
	rm -f bin/spvi_2
	
clean_build_model:
	rm -f build_model.o get_funcs.o extract_field.o move_and_rewards.o util.o
	rm -f bin/build_model 
	rm -f temp_modelOp_dirName.txt