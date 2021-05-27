import numpy as np
from os.path import join

file = "DG3_g200x200x200_r5k_2LpDynObs_v2_vmax5"



def get_vmax(vel_field):
    all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, all_Yi = vel_field

    nt,nx,ny = all_u_mat.shape
    a, nmodes, b, c = all_ui_mat.shape
    assert(a==nt and b==nx and c==ny)
    _, nrzns, _ = all_Yi.shape

    # vel_mat = np.zeros((nrzns,nt,nx,ny))
    vmax_across_rzns= np.zeros((nrzns,))

    for k in range(0,nrzns,50):
        vmax_arr = np.zeros((nt,))
        print("rzn", k)
        for t in range(nt):
            matmul_ui_Yi_t = 0
            matmul_vi_Yi_t = 0
            matmul_phi_i_phi_Yi_t = 0

            for m in range(nmodes):
                matmul_ui_Yi_t += (all_ui_mat[t,m,:,:]*all_Yi[t,k,m])
                matmul_vi_Yi_t += (all_vi_mat[t,m,:,:]*all_Yi[t,k,m])
                
            u_t_k = all_u_mat[t,:,:] + matmul_ui_Yi_t
            v_t_k = all_v_mat[t,:,:] + matmul_vi_Yi_t
            vmax_arr[t] = np.max(((u_t_k**2) + (v_t_k**2))**0.5)
        vmax_across_rzns[k] = np.max(vmax_arr)
    vmax = np.max(vmax_across_rzns)
    return vmax


all_u_mat = np.load(join(file, 'all_u_mat.npy'))
all_v_mat = np.load(join(file,'all_v_mat.npy'))
all_ui_mat = np.load(join(file,'all_ui_mat.npy'))
all_vi_mat = np.load(join(file,'all_vi_mat.npy'))
all_Yi = np.load(join(file,'all_Yi.npy'))
vel_field = [all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, all_Yi]
vel_fnames = ["all_u_mat", "all_v_mat", "all_ui_mat", "all_vi_mat", "all_Yi"]

for i in range(len(vel_field)):
    fname = vel_fnames[i]
    print(fname,".shape: ", vel_field[i].shape, vel_field[i].dtype)
    print(fname,".min, max, mean: ", np.min(vel_field[i]), np.max(vel_field[i]), np.mean(vel_field[i]))
print("vmax = ",get_vmax(vel_field))

mult_factor = 1.39
vel_field[0] = vel_field[0]*mult_factor
vel_field[1] = vel_field[1]*mult_factor
vel_field[2] = vel_field[2]*(mult_factor**0.5)
vel_field[3] = vel_field[3]*(mult_factor**0.5)
vel_field[4] = vel_field[4]*(mult_factor**0.5)


for i in range(len(vel_field)):
    fname = vel_fnames[i]
    print(fname,".shape: ", vel_field[i].shape, vel_field[i].dtype)
    print(fname,".min, max, mean: ", np.min(vel_field[i]), np.max(vel_field[i]), np.mean(vel_field[i]))

print("vmax = ",get_vmax(vel_field))
# # save arrays
# for i in range(len(vel_field)):
#     fname = vel_fnames[i]
#     np.save(join(file,fname), vel_field[i])