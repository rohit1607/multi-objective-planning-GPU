"""
This file is to generate all_Yi_mat with modes corresponding to
realisations in the train_id_list: list of rzns for training.
The output will further be used to construct a transiton model for DP
based on the given realisations.
"""

from utils.custom_functions import read_pickled_File, get_rzn_ids_for_training_and_testing
from utils.setup_grid import setup_grid
from definition import ROOT_DIR
from os.path import join
import numpy as np

# Get train_id_list
rel_path = 'Experiments/104/QL/num_passes_50/QL_Iter_x1/dt_size_2500/ALPHA_0.05/eps_0_0.1'
exp_num_case_dir = join(ROOT_DIR, rel_path)
train_id_list = read_pickled_File(join(exp_num_case_dir, 'train_id_list'))
train_size = len(train_id_list)
print( "len(train_id_list)= ", train_size)


# get all_Yi_mat of required nT
Yi_mat5k_path = join(ROOT_DIR, 'Input_data_files/nT_60/all_Yi.npy')
all_Yi_mat_5kmodes = np.load(Yi_mat5k_path)
nT, original_nrzns, nmodes = all_Yi_mat_5kmodes.shape
assert (all_Yi_mat_5kmodes.shape == (60, 5000, 5)), "Shape Misamatch: CCheck"


# make all_Yi_mat of wanted size:
# Target_all_Yi_mat = np.zeros((nT, train_size, nmodes))
Target_all_Yi_mat = all_Yi_mat_5kmodes[:,train_id_list,:]
print(Target_all_Yi_mat.shape)


#checks
good_to_save = False

for i in range(2):
    rzn = train_id_list[i]
    print("i, rzn: ", i, rzn)
    for t in range(nT):
        for m in range(nmodes):
            if Target_all_Yi_mat[t,i,m] != all_Yi_mat_5kmodes[t,rzn,m]:
                print("SOMETHIGN WRONG")
                break
            else:
                good_to_save =  True

# Manually change name while saving file
if good_to_save == True:
    np.save(join(ROOT_DIR, 'Input_data_files/nT_60/all_Yi_mat_train_id_list_1'), Target_all_Yi_mat )
    np.save(join(ROOT_DIR, 'Input_data_files/nT_60/train_id_list_1'), train_id_list)





# print(len(train_id_list))

# g, xs, ys, X, Y, vel_field_data, nmodes, useful_num_rzns, paths, params, param_str = setup_grid(num_actions =16)
# dt_size =2500
# target_num_rzns = 3000
# f_train_id_list, f_test_id_list,_,_,_=get_rzn_ids_for_training_and_testing(dt_size, target_num_rzns, paths)
# if f_train_id_list == train_id_list:
#     print("yess")

