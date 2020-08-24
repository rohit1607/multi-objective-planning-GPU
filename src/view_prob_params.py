import numpy as np
from os.path import join
ROOT_DIR = "/home/rohit/Documents/Research/e2e_GPU_DP"

prob_params_rel_path = "src/data_modelOutput/energy1/all_jet_g10x10x10_r5/a2x8_i4_j7_ref1/prob_params.npy"
params = np.load(join(ROOT_DIR, prob_params_rel_path))
params = params.reshape((len(params),))

print("PROB_PARAMS:")
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