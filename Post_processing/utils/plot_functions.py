import numpy as np
import matplotlib.pyplot as plt
import math
from definition import c_ni, ROOT_DIR
from utils.custom_functions import createFolder, picklePolicy, read_pickled_File, extract_velocity
from os.path import join
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.colors as colors
import matplotlib.cm as cmx


def action_to_quiver(a):
    vt = a[0]
    theta = a[1]
    vtx = vt * math.cos(theta)
    vty = vt * math.sin(theta)
    return vtx, vty


# def plot_exact_trajectory_set(g, policy, X, Y, vel_field_data, fpath,
#                                                 fname='Trajectories'):

#     # time calculation and state trajectory
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(1, 1, 1)
#     # set grid
#     minor_xticks = np.arange(g.xs[0] - 0.5 * g.dj, g.xs[-1] + 2 * g.dj, g.dj)
#     minor_yticks = np.arange(g.ys[0] - 0.5 * g.di, g.ys[-1] + 2 * g.di, g.di)

#     major_xticks = np.arange(g.xs[0], g.xs[-1] + 2 * g.dj, 5 * g.dj)
#     major_yticks = np.arange(g.ys[0], g.ys[-1] + 2 * g.di, 5 * g.di)

#     ax.set_xticks(minor_xticks, minor=True)
#     ax.set_yticks(minor_yticks, minor=True)
#     ax.set_xticks(major_xticks)
#     ax.set_yticks(major_yticks)

#     ax.grid(which='major', color='#CCCCCC', linestyle='')
#     ax.grid(which='minor', color='#CCCCCC', linestyle='--')
#     st_point= g.start_state
#     plt.scatter(g.xs[st_point[2]], g.ys[g.ni - 1 - st_point[1]], c='g')
#     plt.scatter(g.xs[g.endpos[1]], g.ys[g.ni - 1 - g.endpos[0]], c='r')
#     plt.grid()
#     plt.gca().set_aspect('equal', adjustable='box')

#     plt.quiver(X, Y, vel_field_data[0][0,:,:], vel_field_data[1][0, :, :])

#     nt, nrzns, nmodes = vel_field_data[4].shape #vel_field_data[4] is all_Yi

#     bad_count =0
#     t_list_all=[]
#     t_list_reached=[]
#     G_list=[]
#     traj_list = []

#     for rzn in range(nrzns):
#         g.set_state(g.start_state)
#         dont_plot =False
#         bad_flag = False
#         # t = 0
#         G = 0

#         xtr = []
#         ytr = []

#         t, i, j = g.start_state

#         a = policy[g.current_state()]
#         try:
#             th, ang = a
#         except:
#             print("exception action: ",a)
#             print("at state: ", g.current_state())
#         xtr.append(g.x)
#         ytr.append(g.y)

#         # while (not g.is_terminal()) and g.if_within_actionable_time():
#         while True:
#             vx, vy = extract_velocity(vel_field_data, t, i, j, rzn)
#             r = g.move_exact(a, vx, vy)
#             G = G + r
#             # t += 1
#             (t, i, j) = g.current_state()

#             xtr.append(g.x)
#             ytr.append(g.y)

#             # if edge state encountered, then increment badcount and Dont plot
#             if g.if_edge_state((i,j)):
#                 bad_count += 1
#                 # dont_plot=True
#                 break

#             if (not g.is_terminal()) and  g.if_within_actionable_time():
#                 a = policy[g.current_state()]
#                 try:
#                     th, ang = a
#                 except:
#                     print("exception action: ",a)
#                     print("at state: ", g.current_state())

#             elif g.is_terminal():
#                 break
#             else:
#             #  i.e. not terminal and not in actinable time.
#             # already checked if ternminal or not. If not terminal 
#             # if time reaches nt ie not within actionable time, then increment badcount and Dont plot
#                 bad_count+=1
#                 bad_flag=True
#                 # dont_plot=True
#                 break


#         if dont_plot==False:
#             plt.plot(xtr, ytr)
#             traj_list.append((xtr, ytr))
#             t_list_all.append(t)
#             if bad_flag==False:
#                 t_list_reached.append(t)
#             G_list.append(G)
        
#         #ADDED for trajactory comparison
#         else:
#             traj_list.append(None)
#             t_list_all.append(None)
#             G_list.append(None)

#     if fname != None:
#         plt.savefig(join(fpath,fname), dpi=300)
#         print("*** pickling traj_list ***")
#         picklePolicy(traj_list, join(fpath,fname))
#         print("*** pickled ***")

#     bad_count_tuple = (bad_count, str(bad_count * 100 / nrzns) + '%')
#     return t_list_all, t_list_reached, G_list, bad_count_tuple




def plot_exact_trajectory_set(g, policy, X, Y, vel_field_data, nmodes, train_id_set, test_id_set, goodlist, fpath, fname='Trajectories'):
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

    minor_ticks = [i/100 for i in range(101) if i%20!=0]
    major_ticks = [i/100 for i in range(0,120,20)]

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

    # n_rzn,m,n = vStream_x.shape
    bad_count =0

    t_list=[]
    G_list=[]
    traj_list = []

    for rzn in goodlist:
        # print("rzn: ", rzn)
        color = 'r'
        if rzn in train_id_set:
            color = 'b'
        elif rzn in test_id_set:
            color = 'g'


        g.set_state(g.start_state)
        dont_plot =False
        bad_flag = False
        # t = 0
        G = 0

        xtr = []
        ytr = []

        t, i, j = g.start_state

        a = policy[g.current_state()]

        xtr.append(g.x)
        ytr.append(g.y)
        loop_count = 0
        # while not g.is_terminal() and g.if_within_actionable_time() and g.current_state:
        # print("__CHECK__ t, i, j")
        while True:
            loop_count += 1
            vx, vy = extract_velocity(vel_field_data, t, i, j, rzn)
            # r = g.move_exact(a, vStream_x[rzn, i, j], vStream_y[rzn, i, j])
            r = g.move_exact(a, vx, vy)

            G = G + r
            s = g.current_state()
            (t, i, j) = s

            xtr.append(g.x)
            ytr.append(g.y)

            if g.if_edge_state((i,j)):
                bad_count += 1
                # dont_plot=True
                break

            if (not g.is_terminal(almost=True)) and  g.if_within_actionable_time():
                a = policy[g.current_state()]
            elif g.is_terminal(almost=True):
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
            if color =='g':
                plt.plot(xtr, ytr, color = color, zorder = 1e5)
            else:
                plt.plot(xtr, ytr, color = color)


        # if bad flag is True then append None to the list. These nones are counted later
        if bad_flag == False:  
            traj_list.append((xtr,ytr))
            t_list.append(t)
            G_list.append(G)
        #ADDED for trajactory comparison
        else:
            traj_list.append(None)
            t_list.append(None)
            G_list.append(None)


    if fname != None:

        plt.savefig(join(fpath,fname),bbox_inches = "tight", dpi=200)
        plt.cla()
        plt.close(fig)
        print("*** pickling traj_list ***")
        picklePolicy(traj_list, join(fpath,fname))
        print("*** pickled ***")

    return t_list, G_list, bad_count



def plot_exact_trajectory_set_DP(g, policy, X, Y, vel_field_data, test_id_list, fpath,
                                                fname='Trajectories'):

    # time calculation and state trajectory
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    # set grid
    minor_xticks = np.arange(g.xs[0] - 0.5 * g.dj, g.xs[-1] + 2 * g.dj, g.dj)
    minor_yticks = np.arange(g.ys[0] - 0.5 * g.di, g.ys[-1] + 2 * g.di, g.di)

    major_xticks = np.arange(g.xs[0], g.xs[-1] + 2 * g.dj, 5 * g.dj)
    major_yticks = np.arange(g.ys[0], g.ys[-1] + 2 * g.di, 5 * g.di)

    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks(minor_yticks, minor=True)
    ax.set_xticks(major_xticks)
    ax.set_yticks(major_yticks)

    ax.grid(which='major', color='#CCCCCC', linestyle='')
    ax.grid(which='minor', color='#CCCCCC', linestyle='--')
    st_point= g.start_state
    plt.scatter(g.xs[st_point[2]], g.ys[g.ni - 1 - st_point[1]], c='g')
    plt.scatter(g.xs[g.endpos[1]], g.ys[g.ni - 1 - g.endpos[0]], c='r')
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')

    # plt.quiver(X, Y, vel_field_data[0][0,:,:], vel_field_data[1][0, :, :])

    # nt, nrzns, nmodes = vel_field_data[4].shape #vel_field_data[4] is all_Yi

    bad_count =0
    t_list_all=[]
    t_list_reached=[]
    G_list=[]
    traj_list = []

    for rzn in test_id_list:
        g.set_state(g.start_state)
        dont_plot =False
        bad_flag = False
        # t = 0
        G = 0

        xtr = []
        ytr = []

        t, i, j = g.start_state

        a = policy[g.current_state()]
        xtr.append(g.x)
        ytr.append(g.y)

        # while (not g.is_terminal()) and g.if_within_actionable_time():
        while True:
            vx, vy = extract_velocity(vel_field_data, t, i, j, rzn)
            r = g.move_exact(a, vx, vy)
            G = G + r
            # t += 1
            (t, i, j) = g.current_state()

            xtr.append(g.x)
            ytr.append(g.y)

            # if edge state encountered, then increment badcount and Dont plot
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


        if dont_plot==False:
            plt.plot(xtr, ytr)
           
        if bad_flag==False:
            traj_list.append((xtr, ytr))
            t_list_all.append(t)
            G_list.append(G)
            t_list_reached.append(t)

        
        #ADDED for trajactory comparison
        else:
            traj_list.append(None)
            t_list_all.append(None)
            G_list.append(None)

    if fname != None:
        plt.savefig(join(fpath, fname), dpi=300)
        print("*** pickling traj_list ***")
        picklePolicy(traj_list, join(fpath,fname))
        print("*** pickled ***")

    bad_count_tuple = (bad_count, str(bad_count * 100 / len(test_id_list)) + '%')
    return t_list_all, t_list_reached, G_list, bad_count_tuple



def plot_learned_policy(g, DP_params = None, QL_params = None, vel_field_data = None, showfig = False):
    """
    Plots learned policy
    :param g: grid object
    :param DP_params: [policy, filepath]
    :param QL_params: [policy, Q, init_Q, label_data, filepath]  - details mentioned below
    :param showfig: whether you want to see fig during execution
    :return:
    """
    """
    QL_params:
    :param Q: Leared Q against which policy is plotted. This is needed just for a check in the QL case. TO plot policy only at those states which have been updated
    :param policy: Learned policy.
    :param init_Q: initial value for Q. Just like Q, required only for the QL policy plot
    :param label_data: Labels to put on fig. Currently requiered only for QL
    """
    # TODO: check QL part for this DG3
    # full_file_path = ROOT_DIR
    if DP_params == None and QL_params == None:
        print("Nothing to plot! Enter either DP or QL params !")
        return

    # set grid
    fig1 = plt.figure(figsize=(10, 10))
    ax1 = fig1.add_subplot(1, 1, 1)

    minor_xticks = np.arange(g.xs[0] - 0.5 * g.dj, g.xs[-1] + 2 * g.dj, g.dj)
    minor_yticks = np.arange(g.ys[0] - 0.5 * g.di, g.ys[-1] + 2 * g.di, g.di)

    major_xticks = np.arange(g.xs[0], g.xs[-1] + 2 * g.dj, 5 * g.dj)
    major_yticks = np.arange(g.ys[0], g.ys[-1] + 2 * g.di, 5 * g.di)

    ax1.set_xticks(minor_xticks, minor=True)
    ax1.set_yticks(minor_yticks, minor=True)
    ax1.set_xticks(major_xticks)
    ax1.set_yticks(major_yticks)

    ax1.grid(which='major', color='#CCCCCC', linestyle='')
    ax1.grid(which='minor', color='#CCCCCC', linestyle='--')
    xtr=[]
    ytr=[]
    ax_list=[]
    ay_list=[]


    if QL_params != None:
        policy, Q, init_Q, label_data, full_file_path, fname = QL_params
        F, ALPHA, initq, QIters = label_data
        ax1.text(0.1, 9, 'F=(%s)'%F, fontsize=12)
        ax1.text(0.1, 8, 'ALPHA=(%s)'%ALPHA, fontsize=12)
        ax1.text(0.1, 7, 'initq=(%s)'%initq, fontsize=12)
        ax1.text(0.1, 6, 'QIters=(%s)'%QIters, fontsize=12)
        for s in Q.keys():
            t, i, j = s
            # for a in Q[s].keys():
            # if s[t]%2==0: # to print policy at time t = 0
            a = policy[s]
            if not(Q[s][a] == init_Q/2 or Q[s][a] == init_Q): # to plot policy of only updated states
                # t, i, j = s
                xtr.append(g.xs[j])
                ytr.append(g.ys[g.ni - 1 - i])
                # print("test", s, a_policy)
                ax, ay = action_to_quiver(a)
                ax_list.append(ax)
                ay_list.append(ay)
                    # print(i,j,g.xs[j], g.ys[g.ni - 1 - i], ax, ay)
        

        plt.quiver(xtr, ytr, ax_list, ay_list)
        ax1.scatter(g.xs[g.start_state[2]], g.ys[g.ni - 1 - g.start_state[1]], c='g')
        ax1.scatter(g.xs[g.endpos[1]], g.ys[g.ni - 1 - g.endpos[0]], c='r')

        fig1.savefig(full_file_path + fname +'.png', dpi=150)
        if showfig == True:
            plt.show()
        plt.cla()
        plt.close(fig1)
        



    if DP_params != None:
        policy, full_file_path = DP_params
        policy_plot_folder = createFolder(join(full_file_path,'policy_plots'))

        for tt in range(g.nt-1):
            ax_list =[]
            ay_list = []
            vnetx_list = []
            vnety_list = []
            xtr = []
            ytr = []

            for s in g.ac_state_space(time=tt):
                a = policy[s]
                t, i, j = s
                xtr.append(g.xs[j])
                ytr.append(g.ys[g.ni - 1 - i])
                # print("test", s, a_policy)
                ax, ay = action_to_quiver(a)
                # if you enter vel_field_data, then the "net" "mean" vector will be plotted.
                if vel_field_data != None:
                    vx = vel_field_data[0][t,i,j]
                    vy = vel_field_data[1][t,i,j]
                    vnetx = ax + vx
                    vnety = ay + vy
                    vnetx_list.append(vnetx)
                    vnety_list.append(vnety)
                ax_list.append(ax)
                ay_list.append(ay)

            plt.quiver(xtr, ytr, ax_list, ay_list)
            ax1.scatter(g.xs[g.start_state[2]], g.ys[g.ni - 1 - g.start_state[1] ], c='g')
            ax1.scatter(g.xs[g.endpos[1]], g.ys[g.ni - 1 - g.endpos[0]], c='r')
            if showfig ==True:
                plt.show()
            fig1.savefig(full_file_path + '/policy_plots/policy_plot_t' + str(tt), dpi=150)
            plt.clf()
            fig1.clf()

            if vel_field_data != None:
                plt.quiver(xtr, ytr, vnetx_list, vnety_list)
                ax1.scatter(g.xs[g.start_state[2]], g.ys[g.ni - 1 - g.start_state[1] ], c='g')
                ax1.scatter(g.xs[g.endpos[1]], g.ys[g.ni - 1 - g.endpos[0]], c='r')
                fig1.savefig(full_file_path + '/policy_plots/vnet_plot_t' + str(tt), dpi=150)
                plt.clf()
                fig1.clf()

    return

def plot_and_return_exact_trajectory_set_train_data(g, policy, X, Y, vel_field_data, nmodes, train_id_list, fpath, fname='Trajectories'):
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

    for rzn in train_id_list:
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

            if (not g.is_terminal(almost=True)) and  g.if_within_actionable_time():
                a = policy[g.current_state()]
            elif g.is_terminal(almost=True):
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
            plt.plot(xtr, ytr)

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
        picklePolicy(traj_list, join(fpath,fname))
        picklePolicy(sars_traj_list,join(fpath,'sars_traj_'+fname) )
        print("*** pickled ***")

    return t_list, G_list, bad_count

def plot_max_Qvalues(Q, policy, XP, YP, fpath, fname, showfig = False):
    print("--- in plot_functions.plot_max_Qvalues---")
    # get grid size
    m,n = XP.shape
    Z = np.zeros((m,n))

    for s in Q.keys():
        a = policy[s]
        t,i,j = s
        # i,j =s
        Z[i,j]= Q[s][a]

    fig = plt.figure()
    ax = Axes3D(fig)
    # ax = plt.axes(projection="3d")
    mycmap = plt.get_cmap('coolwarm')

    ax.plot_surface(XP, YP, Z, cmap=mycmap, linewidth=0, antialiased=False)

    if showfig == True:
        plt.show()
    # TODO: see if 3d plot can be saved. low prioirty
    # plt.savefig(join(fpath,fname))


def plot_max_delQs(max_delQ_list_1, filename= None ):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(max_delQ_list_1)

    if filename!= None:
        plt.savefig(filename, dpi=200)
        plt.cla()
    plt.close(fig)

    return


def plot_paths_colored_by_EAT(plotFile=None, baseFile=None, savePath_fname=None):
    msize = 15
    fsize = 3

    #---------------------------- beautify plot ---------------------------
    # time calculation and state trajectory
    fig = plt.figure(figsize=(fsize, fsize))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    # set grid

    minor_ticks = [i for i in range(101) if i % 20 != 0]
    major_ticks = [i for i in range(0, 120, 20)]

    ax.set_xticks(minor_ticks, minor=True)
    ax.set_xticks(major_ticks, minor=False)
    ax.set_yticks(major_ticks, minor=False)
    ax.set_yticks(minor_ticks, minor=True)

    ax.grid(b=True, which='both', color='#CCCCCC', axis='both', linestyle='-', alpha=0.5)
    ax.tick_params(axis='both', which='both', labelsize=6)

    ax.set_xlabel('X (Non-Dim)')
    ax.set_ylabel('Y (Non-Dim)')

    #     st_point= g.start_state
    #     plt.scatter(g.xs[st_point[1]], g.ys[g.ni - 1 - st_point[0]], marker = 'o', s = msize, color = 'k', zorder = 1e5)
    #     plt.scatter(g.xs[g.endpos[1]], g.ys[g.ni - 1 - g.endpos[0]], marker = '*', s = msize*2, color ='k', zorder = 1e5)
    plt.gca().set_aspect('equal', adjustable='box')




    #---------------------------- main plot ---------------------------
    # read file
    plot_set = read_pickled_File(plotFile)

    # calculate time
    time_list = []
    l = len(plot_set)

    # if baseFile is provided, comparison plot will be made. colorbar will show EAT time differnces.
    if baseFile != None:
        base_traj_set = read_pickled_File(baseFile)
        l_base = len(base_traj_set)
        # sanity check
        if l != l_base:
            print("ERROR: Unfair Comparison. Two lists should have data across same number of realisations")
            return

        for i in range(l):
            if plot_set[i] != None and base_traj_set[i] != None:
                t_plot_set_i = len(plot_set[i][0])
                t_base_set_i = len(base_traj_set[i][0])
                time_list.append(t_plot_set_i - t_base_set_i)

    # if baseFile is NOT provided, then the basePlot data will be plotted.
    else:
        for i in range(l):
            if plot_set[i] != None:
                time_list.append(len(plot_set[i][0]))

    # set colormap
    jet = cm = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=np.min(time_list), vmax=np.max(time_list))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    scalarMap._A = []

    # plot plot_set
    for i in range(int(l)):
        if plot_set[i] != None:
            colorval = scalarMap.to_rgba(time_list[i])
            plt.plot(plot_set[i][0], plot_set[i][1], color=colorval, alpha=0.6)
    plt.colorbar(scalarMap)

    if savePath_fname != None:
        plt.savefig(savePath_fname, bbox_inches="tight", dpi=300)

    plt.show()

    return time_list