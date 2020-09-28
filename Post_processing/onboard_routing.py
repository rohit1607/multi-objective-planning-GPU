from utils.setup_grid import setup_grid
from definition import ROOT_DIR
from os.path import join
from utils.custom_functions import read_pickled_File, max_dict, picklePolicy, calc_mean_and_std, writePolicytoFile,stochastic_action_eps_greedy, extract_velocity
from QL.Build_Q_from_Trajs import Q_update
import matplotlib.pyplot as plt
import copy
import numpy as np
import ast
import random
from explict_plot import plot_and_return_exact_trajectory_set_train_data
import argparse
from collections import deque


# def sas_match(query_s1_a_s2, s1_a_r_s2):
#     """
#     returns true if the inputs match.
#     """
#     q_s1, q_a, q_s2 = query_s1_a_s2
#     s1, a, r, s2 = s1_a_r_s2
#     if s1 == q_s1 and s2 == q_s2 and a == q_a:
#         return True
#     else:
#         return False


# def get_similar_sas_traj_ids(query_s1_a_s2, sars_traj_list):
#     """
#     sars_traj_list = [ [(s1_a_s2), (s2_a_s3)..()] , [] , ..  [] ]
#     matched_rzn_id_info = [ (traj_id, transition_id), () .... ()]
#     """
#     matched_rzn_id_info = []
#     for sars_traj_IDX in range(len(sars_traj_list)):
#         sars_traj = sars_traj_list[sars_traj_IDX]
#         if sars_traj != None:
#             for sars_IDX in range(len(sars_traj)):
#                 s1_a_r_s2 = sars_traj[sars_IDX]
#                 if sas_match(query_s1_a_s2, s1_a_r_s2):
#                     # train_id_list[sars_traj_IDX] is the rzn id of the 5k rzns -may not use
#                     # sars_traj_IDX is the trajectory index.
#                     # sars_IDX is the index of a specific sars_traj where the match happened
#                     # so that we dont have to search again while updating Q in future rollout
#                     matched_rzn_id_info.append((sars_traj_IDX, sars_IDX))
#                     break
#     return matched_rzn_id_info

# def update_Q_in_future_kth_rzn(g, Q, N, s2, rzn):
#     """
#     rzn_id_info = (sars_traj_IDX -> idx of sars_traj_list , sars_idx_of_match)
#     """
#     sars_traj_IDX, sars_match_IDX = rzn_id_info
#     sars_traj = sars_traj_list[sars_traj_IDX]
#     max_delQ = 0
#     for k in range(sars_match_IDX, len(sars_traj)):
#         sars = sars_traj[k]
#         Q, N, max_delQ = Q_update(Q, N, max_delQ, sars, ALPHA/10, g, N_inc)
#     return Q, N

# def get_similar_rzn_ids_V2(query_s1_a_s2, g, vel_field_data, nmodes):

#     s1, a, s2 = query_s1_a_s2
#     t, i, j = s1
#     s2_t, s2_i, s2_j = s2
#     matched_rzns = []
#     for rzn in train_id_list:
#         # vx = Vx_rzns[rzn,i,j]
#         # vy = Vy_rzns[rzn,i,j]
#         vx, vy = extract_velocity(vel_field_data, t, i, j, rzn)
#         g.set_state(s1)
#         g.move_exact(a, vx, vy)
#         t2, i2, j2 = g.current_state()
#         if s2_t==t2 and (s2_i == i2 or s2_i == i2 + 1  or s2_i == i2 - 1  or s2_j == j2  or  s2_j == j2 + 1  or s2_j == j2 - 1):
#             matched_rzns.append(rzn)

#     return matched_rzns


def get_similar_rzn_ids(query_cs1_a_s2, g, vel_field_data, nmodes):

    cs1, a, s2 = query_cs1_a_s2
    t, x, y, i, j = cs1
    matched_rzns = set()
    for rzn in train_id_list:
        # vx = Vx_rzns[rzn,i,j]
        # vy = Vy_rzns[rzn,i,j]
        vx, vy = extract_velocity(vel_field_data, t, i, j, rzn)
        g.set_state((t, i, j), xcoord = x, ycoord = y)
        g.move_exact(a, vx, vy)
        # assert(t == s2[0]), "damn"
        if g.current_state() == s2:
            matched_rzns.add(rzn)

    return matched_rzns


def update_Q_in_future_kth_rzn(g, Q, N, vel_field_data, nmodes, s1, rzn, eps):
    """
    almost same as from Run_Q_learning_episode()
    s2: current state in whilie simulating roolout
    """

    t, i, j = s1
    g.set_state(s1)
    dummy_policy = None   #stochastic_action_eps_greedy() here, uses Q. so policy is ingnored anyway
    # a1 = stochastic_action_eps_greedy(policy, s1, g, eps, Q=Q)
    count = 0
    max_delQ = 0

    # while not g.is_terminal() and g.if_within_TD_actionable_time():
    while not g.is_terminal(s1) and not g.if_edge_state(s1) and g.if_within_actionable_time():
        """Will have to change this for general time"""
        
        t, i, j = s1
        a1 = stochastic_action_eps_greedy(dummy_policy, s1, g, eps, Q=Q)
        vx, vy = extract_velocity(vel_field_data, t, i, j, rzn)
        r = g.move_exact(a1, vx, vy, rzn)
        # r = g.move_exact(a1, Vx_rzns[rzn, i, j], Vy_rzns[rzn, i, j])
        s2 = g.current_state()
        # if g.is_terminal() or (not g.if_within_actionable_time()):

        alpha = ALPHA / N[s1][a1]
        N[s1][a1] += N_inc

        #maxQsa = 0 if next state is a terminal state/edgestate/outside actionable time
        max_q_s2_a2= 0
        if not g.is_terminal(s2) and not g.if_edge_state(s2) and g.if_within_actionable_time():
            a2, max_q_s2_a2 = max_dict(Q[s2])

        old_qsa = Q[s1][a1]
        Q[s1][a1] = Q[s1][a1] + alpha*(r + max_q_s2_a2 - Q[s1][a1])

        if np.abs(old_qsa - Q[s1][a1]) > max_delQ:
            max_delQ = np.abs(old_qsa - Q[s1][a1])


        s1 = s2
        # t, i, j = s1

    return Q, N



def update_Q_in_future_rollouts(g, Q, N, cs1as2_list, vel_field_data, nmodes, loop_count):

    #initialise a list of sets
    assert(len(cs1as2_list) == rollout_interval), "check bro"
    set_list = [ ] #list of match_rzn_sets

    # get current state from latest transition. latest items are leftmost in list
    _,_,s2 = cs1as2_list[0]
    check_t, _, _ = s2
    assert(check_t%rollout_interval == 0), "checkkk brooo"

    #build sets for each tranisition
    for cs1_a_s2 in cs1as2_list:
        # match_rzn_set is set of mathcing realisation
        match_rzn_set = get_similar_rzn_ids(cs1_a_s2, g, vel_field_data, nmodes)
        set_list.append(match_rzn_set)

    #find set intersection
    for i in range(1, len(set_list)):
        set_list[0] = set_list[0].intersection(set_list[i])
        print("***** CHECK intetsiction size ***** ", len(set_list[0]))
    intersection_rzns = set_list[0]
    if len(intersection_rzns)<50:
        print("***** CHECK intersection set ", intersection_rzns)

    # update Q,N for rzns in intersection_rzns
    for rzn in intersection_rzns:
        for ntimes in range(num_rollouts):
            Q, N = update_Q_in_future_kth_rzn(g, Q, N, vel_field_data, nmodes, s2, rzn, eps_0)

    return Q, N



def run_and_plot_onboard_routing_episodes(setup_grid_params, Q, N, fpath, fname):
# g, xs, ys, X, Y, vel_field_data, nmodes, useful_num_rzns, paths, params, param_str
    g, xs, ys, X, Y, vel_field_data, nmodes, _, paths, _, _ = setup_grid_params
    g.make_bcrumb_dict(paths, train_id_list)
   
    gcopy = copy.deepcopy(g)
    # Copy Q to Qcopy

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

    # plt.quiver(X, Y, Vx_rzns[0, :, :], Vy_rzns[0, :, :])

    
    t_list=[]
    traj_list = []
    bad_count = 0
    # for k in range(len(test_id_list)):
    for k in range(n_test_paths_range[0], n_test_paths_range[1]):
        Qcopy = copy.deepcopy(Q)
        Ncopy = copy.deepcopy(N)
        rzn = test_id_list[k]

        init_list = [None for i in range(rollout_interval)]
        cs1as2_list = deque(init_list)         #to keep a fixed lenght list representation  

        print("-------- In rzn ", rzn, " of test_id_list ---------")
        g.set_state(g.start_state)
        dont_plot =False
        bad_flag = False

        xtr = []
        ytr = []

        s1 = g.start_state
        t, i, j = s1
        cs1 = (t, g.x, g.y ,i, j)
        a, q_s_a = max_dict(Qcopy[s1])

        xtr.append(g.x)
        ytr.append(g.y)
        loop_count = 0
        # while not g.is_terminal() and g.if_within_actionable_time() and g.current_state:
        # print("__CHECK__ t, i, j")
        while True:
            loop_count += 1
            vx, vy = extract_velocity(vel_field_data, t, i, j, rzn)
            r = g.move_exact(a, vx, vy)
            # r = g.move_exact(a, Vx_rzns[rzn, i, j], Vy_rzns[rzn, i, j])
            s2 = g.current_state()
            (t, i, j) = s2
            cs1_a_s2 = (cs1, a, s2)

            # keep n latest transitions where n = rollout_interval
            cs1as2_list.pop()
            cs1as2_list.appendleft(cs1_a_s2)

            xtr.append(g.x)
            ytr.append(g.y)


            if g.if_edge_state((i,j)):
                bad_count += 1
                # dont_plot=True
                break
            if (not g.is_terminal(almost = True)) and  g.if_within_actionable_time():
                if loop_count % rollout_interval == 0:
                    print("------------loopcount/mission_time =", loop_count)
                    # for kk in range(len(cs1as2_list)):
                    #     check_cs1_a_s2 = cs1as2_list[kk]
                    #     check_cs1 = check_cs1_a_s2[0]
                    #     check_s2 = check_cs1_a_s2[2]
                    #     tij1 = (check_cs1[0],check_cs1[3],check_cs1[4])
                        # print("check: ", tij1, check_s2)
                    Qcopy, Ncopy = update_Q_in_future_rollouts(gcopy, Qcopy, Ncopy, cs1as2_list, vel_field_data, nmodes, loop_count)
                s1 = s2 #for next iteration of loop
                cs1 = (t, g.x, g.y, i, j)
                a, q_s_a = max_dict(Qcopy[s1])
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


        if dont_plot==False:
            plt.plot(xtr, ytr)
        # if bad flag is True then append None to the list. These nones are counted later
        if bad_flag == False:  
            traj_list.append((xtr,ytr))
            t_list.append(t)
        #ADDED for trajactory comparison
        else:
            traj_list.append(None)
            t_list.append(None)


    if fname != None:
        plt.savefig(join(fpath,fname),bbox_inches = "tight", dpi=200)
        plt.cla()
        plt.close(fig)
        writePolicytoFile(t_list, join(fpath,fname+'tlist' ))
        picklePolicy(traj_list, join(fpath,fname+'_coord_traj'))
        print("*** pickled phase2 traj_list ***")

    return t_list, bad_count



def run_onboard_routing_for_test_data(exp_num_case_dir, setup_grid_params, opfname):
   
    global ALPHA
    global N_inc
    global train_id_list
    global test_id_list
    global sars_traj_list
    global eps_0

    Q = read_pickled_File(join(exp_num_case_dir, 'Q2'))
    N = read_pickled_File(join(exp_num_case_dir, 'N2'))
    policy = read_pickled_File(join(exp_num_case_dir, 'Policy_02'))
    test_id_list = read_pickled_File(join(exp_num_case_dir, 'test_id_list'))
    train_id_list = read_pickled_File(join(exp_num_case_dir, 'train_id_list'))
    # sars_traj_list = read_pickled_File(join(exp_num_case_dir, 'sars_traj_Train_Trajectories_after_exp'))
    train_output_params = read_pickled_File(join(exp_num_case_dir, 'output_paramaters'))

    print("*********** 1 **************\n")
    ALPHA = train_output_params[9]
    N_inc = train_output_params[11]
    eps_0 = 0.05

    print("ALPHA, N_inc = ", ALPHA, N_inc)
    # print('len(sars_traj_list) = ', len(sars_traj_list))
    print("len(train_id_list)= ", len(train_id_list))
    print("len(test_id_list)= ", len(test_id_list))
    print('n_test_paths_range = ', n_test_paths_range)
    t_list, bad_count = run_and_plot_onboard_routing_episodes(setup_grid_params, Q, N,
                                              exp_num_case_dir, opfname )

    print("*********** 2 **************\n")

    phase2_results = calc_mean_and_std(t_list)
    picklePolicy(phase2_results,join(exp_num_case_dir, opfname))
    writePolicytoFile(phase2_results,join(exp_num_case_dir, opfname))
    avg_time_ph2, std_time_ph2, cnt_ph2 , none_cnt_ph2 = phase2_results

    print("test_id_list[range] = ", test_id_list[n_test_paths_range[0]:n_test_paths_range[1]])
    print("----- phase 2 data ---------")
    print("avg_time_ph2", avg_time_ph2,'\n', 
           "std_time_ph2", std_time_ph2, '\n',
            "cnt_ph2",cnt_ph2 , '\n',
           "none_cnt_ph2", none_cnt_ph2)


    # Compare
    g, xs, ys, X, Y, vel_field_data, nmodes, _, _, _, _ = setup_grid_params
    p1_t_list, p1_G_list, p1_bad_count = plot_and_return_exact_trajectory_set_train_data(g, policy, X, Y, 
                                            vel_field_data, nmodes, test_id_list,n_test_paths_range, 
                                            exp_num_case_dir, fname='Explicit_Phase_1_test_' + str(run_number))
    phase1_results = calc_mean_and_std(p1_t_list)
    avg_time_ph1, std_time_ph1, cnt_ph1 , none_cnt_ph1 = phase1_results
    print("----- phase 1 data ---------")
    print("avg_time_ph1", avg_time_ph1,'\n', 
           "std_time_ph1", std_time_ph1, '\n',
            "cnt_ph1",cnt_ph1 , '\n',
           "none_cnt_ph1", none_cnt_ph1)


    
    return




parser = argparse.ArgumentParser(description='Take parameters as input args.')

parser.add_argument('n_start', type=int, help='start of n_test_paths_range')
parser.add_argument('n_end', type=int, help='start of n_test_paths_range')

args = parser.parse_args()


global n_test_paths_range
global run_number
global num_rollouts
global rollout_interval


n_start = args.n_start
n_end = args.n_end
# n_test_paths_range = [250, 500]
n_test_paths_range = [n_start, n_end]

rollout_interval= 4
num_rollouts = 1

# run_number = 10000 + n_start
run_number = str(rollout_interval) + '_' + str(num_rollouts)
setup_grid_params = setup_grid(num_actions=16)
opfname = 'phase_2_test_' + str(run_number)
# rel_path = 'Experiments/55/QL/num_passes_50/QL_Iter_x1/dt_size_2000/ALPHA_0.05/eps_0_0.1'
# rel_path = 'Experiments/104/QL/num_passes_50/QL_Iter_x1/dt_size_2500/ALPHA_0.05/eps_0_0.1'
# rel_path = 'Experiments/95/QL/num_passes_50/QL_Iter_x1/dt_size_2500/ALPHA_0.05/eps_0_0.1'
# rel_path = 'Experiments/111/QL/num_passes_50/QL_Iter_x1/dt_size_2500/ALPHA_0.05/eps_0_0.1'
# rel_path = 'Experiments/118/QL/num_passes_50/QL_Iter_x1/dt_size_4000/ALPHA_0.05/eps_0_0.1'

# rel_path = 'Experiments/126/QL/num_passes_50/QL_Iter_x1/dt_size_5000/ALPHA_0.05/eps_0_0.1'
rel_path = 'Experiments/128/QL/num_passes_50/QL_Iter_x1/dt_size_5000/ALPHA_0.05/eps_0_0.1'
exp_num_case_dir = join(ROOT_DIR, rel_path)

run_onboard_routing_for_test_data(exp_num_case_dir, setup_grid_params, opfname)


# test_id_list = read_pickled_File(join(exp_num_case_dir, 'test_id_list'))
# tlist_file = join(exp_num_case_dir, 'TrajTimes2.txt')
# with open(tlist_file, 'r') as f:
#     mylist = ast.literal_eval(f.read())
# n = n_test_paths
# summ = 0
# cnt = 0
# print(len(tlist))
# print(test_id_list[:n])
# for i in range(n):
#     rzn = test_id_list[i]
#     t =  tlist[rzn]
#     print(t)
#     if t != None:
#         summ += tlist[rzn]
#         cnt += 1

# print("mean= ", summ/cnt)
# print("cnt = ", cnt)
# print("pfail or badcount% = ", cnt/n)

