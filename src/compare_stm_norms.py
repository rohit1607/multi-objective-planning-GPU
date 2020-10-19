import numpy as np
import matplotlib.pyplot as plt
file2 = 'data_modelOutput/time/AF_DG_g100x100x60_r2k_spviTimeTest/a1x16_i80_j80_ref2/'
file3 = 'data_modelOutput/time/AF_DG_g100x100x60_r3k_spviTimeTest/a1x16_i80_j80_ref2/'
# file4 = 'data_modelOutput/time/AF_DG_g100x100x60_r4k_spviTimeTest/a1x16_i80_j80_ref2/'
file5 = 'data_modelOutput/time/AF_DG_g100x100x60_r5k_spviTimeTest/a1x16_i80_j80_ref2/'

files = [file2, file3, file5]
l = len(files)
R_list = []
P_val_list = []
for i in range(l):
    R_list.append(np.load(files[i]+"master_R.npy"))
    P_val_list.append(np.load(files[i]+"master_cooVal.npy"))


for P in P_val_list:
    print(P.shape)


for i in range(len(R_list)-1):
    print(i)
    r2 = R_list[i+1]
    r1 = R_list[i]

    delr = np.abs(r2 - r1)
    perc_delr = np.true_divide(delr,1000)

    x = np.arange(len(r1))
    y = perc_delr[:,0]
    plt.plot(x,y)

    plt.title("Convergence of R")
    # plt.savefig("perc_diff_Rs" +str(i) + ".png")
    # plt.clf()


plt.savefig("normalised_diff_Rs.png")





count = 0
# for i in range(len(delR)):
#     den = np.min((np.abs(R1[i]), np.abs(R2[i])))
#     perc_diff = delR[i]/den
#     if perc_diff > 0.01:
#         print(i, delR[i], R1[i], R2[i])
#         count += 1