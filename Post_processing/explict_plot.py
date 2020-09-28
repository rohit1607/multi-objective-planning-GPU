import numpy as np
import matplotlib.pyplot as plt
import math
from definition import c_ni, ROOT_DIR
from utils.custom_functions import createFolder, picklePolicy, read_pickled_File, extract_velocity, calc_mean_and_std
from os.path import join
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.colors as colors
import matplotlib.cm as cmx
from utils.setup_grid import setup_grid
import ast

def plot_and_return_exact_trajectory_set_train_data(g, policy, X, Y, vel_field_data, nmodes, test_id_list, n_test_paths_range, fpath, fname='Trajectories'):
    """
    Makes plots across all rzns with different colors for test and train data
    returns list for all rzns.
    """

    # time calculation and state trajectory
    print("--- in plot_functions.plot_exact_trajectory_set---")

    msize = 15
    # fsize = 3

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)

    minor_ticks = [i/100.0 for i in range(101) if i%20!=0]
    major_ticks = [i/100.0 for i in range(0,120,20)]

    ax.set_xticks(minor_ticks, minor=True)
    ax.set_xticks(major_ticks, minor=False)
    ax.set_yticks(major_ticks, minor=False)
    ax.set_yticks(minor_ticks, minor=True)

    ax.grid(b= True, which='both', color='#CCCCCC', axis='both',linestyle = '-', alpha = 0.5)
    ax.tick_params(axis='both', which='both', labelsize=6)

    ax.set_xlabel('X (Non-Dim)')
    ax.set_ylabel('Y (Non-Dim)')

    st_point= g.start_state
    plt.scatter(g.xs[st_point[1]], g.ys[g.ni - 1 - st_point[0]], marker = 'o', s = msize, color = 'k', zorder = 1e5)
    plt.scatter(g.xs[g.endpos[1]], g.ys[g.ni - 1 - g.endpos[0]], marker = '*', s = msize*2, color ='k', zorder = 1e5)
    plt.gca().set_aspect('equal', adjustable='box')

    # plt.quiver(X, Y, vStream_x[0, :, :], vStream_y[0, :, :])

    # _,m,n = vStream_x.shape
    bad_count =0

    t_list=[]
    G_list=[]
    traj_list = []
    sars_traj_list = []

    for rzn in test_id_list[n_test_paths_range[0]:n_test_paths_range[1]]:
        # print("rzn: ", rzn)

        g.set_state(g.start_state)
        dont_plot =False
        bad_flag = False
        # t = 0
        G = 0

        xtr = []
        ytr = []
        sars_traj = []

        s1 = g.start_state
        t, i, j = s1

        a = policy[g.current_state()]

        xtr.append(g.x)
        ytr.append(g.y)
        loop_count = 0
        # while not g.is_terminal() and g.if_within_actionable_time() and g.current_state:
        # print("__CHECK__ t, i, j")
        while True:
            loop_count += 1
            vx, vy = extract_velocity(vel_field_data, t, i, j, rzn)
            r = g.move_exact(a, vx, vy)

            G = G + r
            s2 = g.current_state()
            (t, i, j) = s2

            sars_traj.append((s1, a, r, s2))
            xtr.append(g.x)
            ytr.append(g.y)

            s1 = s2 #for next iteration of loop
            if g.if_edge_state((i,j)):
                bad_count += 1
                # dont_plot=True
                break

            if (not g.is_terminal(almost = True)) and  g.if_within_actionable_time():
                a = policy[g.current_state()]
            elif g.is_terminal(almost = True):
                break
            else:
            #  i.e. not terminal and not in actinable time.
            # already checked if ternminal or not. If not terminal 
            # if time reaches nt ie not within actionable time, then increment badcount and Dont plot
                bad_count+=1
                bad_flag=True
                # dont_plot=True
                break

            #Debugging measure: additional check to break loop because code gets stuck sometims
            if loop_count > g.ni * c_ni:
                print("t: ", t)
                print("g.current_state: ", g.current_state())
                print("xtr: ",xtr)
                print("ytr: ",ytr)
                break

            # if t > g.ni * c_ni: #if trajectory goes haywire, dont plot it.
            #     bad_count+=1
            #     dont_plot=True
            #     break

        if dont_plot==False:
            plt.plot(xtr, ytr, '--')

        # if bad flag is True then append None to the list. These nones are counted later
        if bad_flag == False:  
            sars_traj_list.append(sars_traj) 
            traj_list.append((xtr,ytr))
            t_list.append(t)
            G_list.append(G)
        #ADDED for trajactory comparison
        else:
            sars_traj_list.append(None) 
            traj_list.append(None)
            t_list.append(None)
            G_list.append(None)


    if fname != None:
        plt.savefig(join(fpath,fname),bbox_inches = "tight", dpi=200)
        plt.cla()
        plt.close(fig)
        print("*** pickling traj_list ***")
        picklePolicy(traj_list, join(fpath,fname + 'coord_traj'))
        # picklePolicy(sars_traj_list,join(fpath,'sars_traj_'+fname) )
        print("*** pickled ***")

    return t_list, G_list, bad_count


# g, xs, ys, X, Y, vel_field_data, nmodes, useful_num_rzns, paths, params, param_str = setup_grid(num_actions=16)

# rel_path = 'Experiments/26/DP'
# exp_num_case_dir = join(ROOT_DIR, rel_path)
# policy = read_pickled_File(join(exp_num_case_dir, 'policy'))

# # tlist_file = join(exp_num_case_dir, 'TrajTimes2.txt')
# # with open(tlist_file, 'r') as f:
# #     phase1_tlist = ast.literal_eval(f.read())

# test_id_rel_path ='Experiments/104/QL/num_passes_50/QL_Iter_x1/dt_size_2500/ALPHA_0.05/eps_0_0.1'
# test_id_list = read_pickled_File(join(test_id_rel_path, 'test_id_list'))

# global n_test_paths_range
# n_test_paths_range = [0, len(test_id_list)]

# t_list, G_list, bad_count = plot_and_return_exact_trajectory_set_train_data(g, policy, X, Y, vel_field_data, nmodes, test_id_list, n_test_paths_range, exp_num_case_dir, fname='Explicit_plot_DPpolicy_104_testid')

# phase1_results = calc_mean_and_std(t_list)
# avg_time_ph1, std_time_ph1, cnt_ph1 , none_cnt_ph1 = phase1_results
# print("avg_time_ph1", avg_time_ph1,'\n', 
#         "std_time_ph1", std_time_ph1, '\n',
#         "cnt_ph1",cnt_ph1 , '\n',
#         "none_cnt_ph1", none_cnt_ph1)







# print(t_list)
# print("stats from explicit plot: ")
# print(calc_mean_and_std(t_list))



# summ = 0
# cnt = 0
# print(len(phase1_tlist))
# print(test_id_list[:n_test_paths])
# for i in range(n_test_paths):
#     rzn = test_id_list[i]
#     t =  phase1_tlist[rzn]
#     print(t)
#     if t != None:
#         summ += phase1_tlist[rzn]
#         cnt += 1
# print("----- phase 1 data and explict plot -----")
# print('n_test_paths= ',n_test_paths)
# print("mean= ", summ/cnt)
# print("cnt = ", cnt)
# print("pfail or badcount% = ", cnt/n_test_paths)