import os
# Project root
ROOT_DIR = "/gdata1/rohitc/e2e_GPU_DP"
print(ROOT_DIR)
# increment N[s][a] by N_inc each time (s,a) visited
N_inc = 0.005

#Take every nth point from a trajectory waypoint list
Sampling_interval = 40

# multiplication factor: t > g.ni * c_ni
c_ni = 1.5

# multiplication factor for sin(theta) in r2 in const_rew_dt in custom_funcions.py
c_r2 = 1

# /gdata1/rohitc/e2e_GPU_DP/src
# /gdata1/rohitc/e2e_GPU_DP/src/data_modelOutput/custom1/AF_DG_g50x50x50_r100_DynObs/a2x16_i25_j5_ref2/0.000000/prob_params.npy