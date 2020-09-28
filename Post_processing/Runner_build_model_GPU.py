
# from utils.build_model_GPU import build_sparse_transition_model
# from utils.build_model import Build_Model
import time
from definition import ROOT_DIR
import numpy as np
from os.path import join
# print("debugging 89th step crash. test 6: trying 8 actions ")
# result : crashes in 97th step.
# print("debugging  contd: test 7: trying reduced xs,ys. gsize is 10 by making Test_grid = True")
# result: works. does not crash.
# print("debugging  contd: test 8s: test 7  +   move result_gpu_list out of loop")
# result: works. does not crash. No observable change in GPU memory usage.
# print("debugging  contd: test 8b: move result_gpu_list out of loop, 8 actions")
# result : crashes in 97th step. just like test 6.
# print("debugging  contd: test 8m: move result_gpu_list out of loop, 8 actions. gsise = 85 with Test_grid = True")
# result : crashes in 65th step.
# print("debugging  contd: test 9: fixed sp_id issue in global kernel. because of not copying from latest commit. ")
# result : works. no crash. problem was missing sp_id. in copying from last to latest comit of Highway. 

# print("Now, debugging wrong traj_set plot. test on small grid of gsize 10, GPU_sgrid_1_")
# result: incorrect policy. start state has policy None! Possible reason is small dt wrt length scale of grid
# print("debugging contd: GPU_sgrid_2_: double dt. now dt = 40e-5")
# result: same error as  prev. snews in trans_dicts are wrong. (0, 7, 5 -> 1, 0, 5)

# print("debugging contd: GPU_test_10: created copy from latest Hw2d commit. test for nt = 5 on full space")
# result: didnt crash. snew seems correct considering dt is small.

# print("debugging contd: GPU_test_11: created copy from latest Hw2d commit. test for nt = 5 on full space")
# result: trajs starting nearby (2-4 steps away) reach target. next, try with further targets.

# print("debugging contd: GPU_test_12: created copy from latest Hw2d commit. test for nt = 30 on full space, dt = 20e-5")
# result: 

# print("debugging contd: GPU_test_13: created copy from latest Hw2d commit. test for nt = 30 on full space, dt = 20e-5")

# print("debugging contd: GPU_test_14: test for nt = 30 on full space, dt = 20e-5, rew = -dt")

# print("debugging contd: GPU_test_15: test for nt = 30 on full space, dt = 40e-5, rew = -dt")
# result: results for test 12-15 in exp_summary file. seem to get better results for dt =40e-5

# print("debugging contd: GPU_test_16: test for nt = 30, nT= 60(at data level) on full space, dt = 40e-5, rew = def")
#Come back to this later after rechecking tests 13 and 15 with correct xs, ys

# print("debugging contd: GPU_test_13b: 12-15 used 120_nT folder and had setupgrid(test=true) call.. test for nt = 30 on full space, dt = 40e-5")
#reulsts: similar. 

# print("debugginh cotd: GPU_test_17: using nT_60 folder. removed wanton setupgrid(), ")
# results: succesful

t1 = time.time()
# build_sparse_transition_model(filename = 'GPU_test_7_', n_actions = 16, nt = 3 )
# build_sparse_transition_model(filename = 'GPU_test_6_', n_actions = 8)
# build_sparse_transition_model(filename = 'GPU_test_8s_', Test_grid = True)
# build_sparse_transition_model(filename = 'GPU_test_8m_', n_actions = 8, Test_grid = True)
# build_sparse_transition_model(filename = 'GPU_test_9_', n_actions = 16)
# build_sparse_transition_model(filename = 'GPU_sgrid_1_', n_actions = 16, Test_grid = True)
# build_sparse_transition_model(filename = 'GPU_sgrid_2_', n_actions = 16, dt = 40e-5, Test_grid = True)
# build_sparse_transition_model(filename = 'GPU_test_10_', n_actions = 16, nt = 5 )
# build_sparse_transition_model(filename = 'GPU_test_11_', n_actions = 16, nt = 5 )
# build_sparse_transition_model(filename = 'GPU_test_12_', n_actions = 16, nt = 30 )
# build_sparse_transition_model(filename = 'GPU_test_13_', n_actions = 16, nt = 30 )
# build_sparse_transition_model(filename = 'GPU_test_15_', n_actions = 16, nt = 30 )
# build_sparse_transition_model(filename = 'GPU_test_16_', n_actions = 16)
# build_sparse_transition_model(filename = 'GPU_test_13b_', n_actions = 16, nt = 30 )


# build_sparse_transition_model(filename = 'GPU_test_17_', n_actions = 16)

from utils.build_model_GPU_for_rzn_list import build_sparse_transition_model
all_Yi = np.load(join(ROOT_DIR, 'Input_data_files/nT_60/all_Yi_mat_train_id_list_1.npy'))
build_sparse_transition_model(filename = 'DG_model_2500_train_id_list_1_',all_Yi=all_Yi, wanted_nrzns = 2500, n_actions = 16)






# Build_Model(filename='CPU_testGrid_3_',n_actions=8, Test_grid=True)

t2 = time.time()

print("Finished executing Runner_build_model_GPU.py")