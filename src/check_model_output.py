import numpy as np
from os.path import join

path = "data_modelOutput/energy2/all_jet_rot1_g70x70x5_r5k_sf1/a2x8_i2_j7_ref1"

files = ["DP_relv_params.npy", "master_cooS1.npy", "master_cooS2.npy", 
            "master_cooVal.npy", "master_R.npy", "prob_params.npy"]

for file in files:
    a = np.load(join(path,file))
    print(file, a.shape)


params = a.reshape((len(a),))
print("\nPROB_PARAMS:")
print(
"gsize  = \t",        params [0], "\n",
"num_actions    = \t",  params [1], "\n", 
"nrzns    = \t",        params [2], "\n",
"F    = \t",            params [3], "\n",
"dt    = \t",           params [4], "\n",
"r_outbound    = \t",   params [5], "\n",
"r_terminal    = \t",   params [6], "\n",
"nmodes    = \t",       params [7], "\n",
"i_term    = \t",       params [8], "\n",
"j_term    = \t",       params [9], "\n",
"nt    = \t",           params [10], "\n",
"is_stationary    = \t",        params [11], "\n",
"term_subgrid_size    = \t",    params [12], "\n",
"reward_type    = \t",      params [13], "\n",
"num_ac_speeds    = \t",    params [14], "\n",
"num_ac_angles    = \t",    params [15], "\n",
"dx    = \t",               params [16], "\n",
"dy    = \t",               params [17], "\n",
)

print()
gsize = params[0]
nt = params[10]
num_actions = params[1]
n_states = gsize*gsize*nt
print("n_states= ", n_states)

concated_R_size = n_states*num_actions
print("concated_R_size= ", concated_R_size)

DP_relv_params = np.load(join(path,"DP_relv_params.npy"))
print("DP_relv_params = \n", DP_relv_params)