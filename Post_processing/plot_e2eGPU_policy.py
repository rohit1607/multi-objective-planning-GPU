import numpy as np
from utils.setup_grid import setup_grid
from definition import ROOT_DIR
from os.path import join
from utils.custom_functions import extract_velocity, get_angle_in_0_2pi
import matplotlib.pyplot as plt
import math


# plot function
def action_to_quiver(a):
    vt = a[0]
    theta = a[1]
    vtx = vt * math.cos(theta)
    vty = vt * math.sin(theta)
    return vtx, vty


def s1_id(state_tuple):
    t, i, j = state_tuple
    return j + gsize*i + (gsize**2)*t


def f(frac):
    if frac < 0:
        return -1
    else:
        return 1
    

def get_cellcenter_from_state(state, g):
    t,i,j = state
    x0 = g.xs[j]
    y0 = g.ys[g.ni - 1 - i]
    return x0, y0

def bilinear_interpolate(x, y, x_tuple, y_tuple, ordered_vals):
    # https://en.wikipedia.org/wiki/Bilinear_interpolation
    # returns bilienarly interpolated value
    v11, v12, v21, v22 = ordered_vals
    x1, x2 = x_tuple
    y1, y2 = y_tuple

    denom = 1/((x2-x1)*(y2-y1))
    assert(x2!=x1), "DIVISION BY ZERO, equality in x"
    assert(y2!=y1), "DIVISION BY ZERO, equality in y"
    x_vec = np.array([x2-x, x-x1]).reshape((1,2))
    val_mat =  np.array([[v11, v12],
                         [v21, v22]])
    y_vec = np.array([y2-y, y-y1]).reshape((2,1))

    v = denom * np.matmul(x_vec, np.matmul(val_mat,y_vec))
    # v.shape = (1,1)

    return v[0][0]



def traingle_interpolate(x, y, p_list, v_list):
    # https://codeplea.com/triangular-interpolation
    (x1, y1), (x2, y2), (x3, y3) = p_list
    V = np.array(v_list)
    W = np.zeros((3,))
    denom = (y2-y3)*(x1-x3) + (x3-x2)*(y1-y3)
    W[0] = (y2-y3)*(x-x3) + (x3-x2)*(y-y3)
    W[1] = (y3-y1)*(x-x3) + (x1-x3)*(y-y3)
    W[0] = W[0]/denom
    W[1] = W[1]/denom
    W[2] = 1 - W[0] - W[1]
    v = np.dot(V,W)
    # v in a float
    return v



def interpolate_as_per_term_status(x, y, x_tuple, y_tuple, vx_tuple, vy_tuple, method = None):
    # interpolate based on apt method according to term_status
    vx11, vx12, vx21, vx22 = vx_tuple
    vy11, vy12, vy21, vy22 = vy_tuple
    x1, x2 = x_tuple
    y1, y2 = y_tuple

    if method == 'bilinear':
        itp_vx = bilinear_interpolate(x, y, x_tuple, y_tuple, vx_tuple)
        itp_vy = bilinear_interpolate(x, y, x_tuple, y_tuple, vy_tuple)

    elif method == 'triangle':
        p_list = [(x1, y1), (x1, y2), (x2, y1)]
        vx_list = [vx11, vx12, vx21]
        vy_list = [vy11, vy12, vy21]
        itp_vx = traingle_interpolate(x, y, p_list, vx_list)
        itp_vy = traingle_interpolate(x, y, p_list, vy_list)
    
    else:
        print("no such interpolation method")
        return None

    itp_theta = get_angle_in_0_2pi(math.atan2(itp_vy, itp_vx))
    itp_v = (itp_vx**2 + itp_vy**2)**0.5

    return itp_v, itp_theta 


def interpolate_action_angles(supports, term_status, policy_1d, g):
    F11, th11 = get_action_from_policy1d(policy_1d, supports[0], g) 
    F12, th12 = get_action_from_policy1d(policy_1d, supports[1], g)
    F21, th21 = get_action_from_policy1d(policy_1d, supports[2], g)
    F22, th22 = get_action_from_policy1d(policy_1d, supports[3], g)

    x11, y11 = get_cellcenter_from_state( supports[0], g)
    x12, y12 = get_cellcenter_from_state( supports[1], g)
    x21, y21 = get_cellcenter_from_state( supports[2], g)
    x22, y22 = get_cellcenter_from_state( supports[3], g)

    # assertions for checking for uniform grid points
    assert(x11 == x12)
    assert(x21 == x22)
    assert(y11 == y21)
    assert(y12 == y22)

    x1, y1 = x11, y11
    x2, y2 = x22, y22
    x, y = g.x, g.y
    vx11, vx12, vx21, vx22 = F11*math.cos(th11), F12*math.cos(th12), F21*math.cos(th21), F22*math.cos(th22)
    vy11, vy12, vy21, vy22 = F11*math.sin(th11), F12*math.sin(th12), F21*math.sin(th21), F22*math.sin(th22)

    # default method is bilinear excepet when term_status = [0,0,0,1] (corner of square)
    method = 'bilinear'
    if term_status == [0, 0, 0, 0]:
        pass

    elif term_status in [[0 ,0, 1, 1], [0, 0, 1, 0]]:
        # effective linear interp along y by making values over terminal subrid same as adjacent cells
        # if x,y is in a cell quadrant, such that terminal subgrid is on the the immediate left or right
        vx21 = vx11
        vx22 = vx12
        vy21 = vy11
        vy22 = vy12

    elif term_status in [[0, 1, 0, 1], [0, 1, 0, 0]]:
        # effective linear interp along x by making values over terminal subrid same as adjacent cells
        # if x,y is in a cell quadrant, such that terminal subgrid is on the the immediate top or bottom
        vx12 = vx11
        vx22 = vx21
        vy12 = vy11
        vy22 = vy21

    elif term_status == [0, 0, 0, 1]:
        # if x,y is in a cell quadrant, is at the outside corners of terminal subgrid
        method = 'triangle'

    else:
        raise NameError(" something wrong. this error should not occur") 
        #these term status are not possible to encounter
        
    x_tuple = (x1, x2)
    y_tuple = (y1, y2)
    vx_tuple = (vx11, vx12, vx21, vx22)
    vy_tuple = (vy11, vy12, vy21, vy22)
    itp_v, itp_theta = interpolate_as_per_term_status(x, y, x_tuple, y_tuple ,vx_tuple ,vy_tuple, method = method)

    return itp_v, itp_theta



def get_interpolated_action(policy_1d, g):
    # F, _ = get_action_from_policy1d(policy_1d, g.current_state(), g)
    t,i,j = g.current_state()
    # x0 = g.xs[j]
    # y0 = g.ys[g.ni - 1 - i]
    x0, y0 = get_cellcenter_from_state(g.current_state(), g)
    delx = g.x - x0
    dely = g.y - y0

    assert(np.abs(delx) <= g.dx/2 or np.abs(dely) <= g.dy/2), "(x,y) , (i,j) mismatch"

    supports = [(t,i,j)]
    supports.append((t, i - f(dely), j))
    supports.append((t, i , j + f(delx)))
    supports.append((t, i - f(dely), j + f(delx)))
    # print("supports= ", supports)

    term_status = [0,0,0,0]
    for i in range(len(term_status)):
        if g.is_terminal(supports[i]):
            term_status[i] = 1
    # print("term_status= ", term_status)
    itp_v, itp_angle = interpolate_action_angles(supports, term_status, policy_1d, g)
     
    return (itp_v, itp_angle)



def get_sg_square_corners(g, position):
    # returns the xy coords for corners of the square that
    # makes up the terminal subgrid
    if position == 'target':
        i,j = g.endpos
    elif position == 'start':
        i,j = g.startpos
    else:
        raise NameError("no such position defined in funcion")

    xf, yf = get_cellcenter_from_state((0,i,j), g)
    x_l = xf - (g.dx/2)
    x_r = x_l +  (g.tsg_size * g.dx)
    y_t = yf + (g.dy/2)
    y_b = y_t - (g.tsg_size * g.dy)

    x_tsg = [x_l, x_r, x_r, x_l]
    y_tsg = [y_t, y_t, y_b, y_b]
    return x_tsg, y_tsg


def get_general_square_corners(g, state):
    t,i,j = state
    xf, yf = get_cellcenter_from_state((t,i,j), g)
    x_l = xf - (g.dx/2)
    x_r = x_l + g.dx
    y_t = yf + (g.dy/2)
    y_b = y_t - g.dy
    x_tsg = [x_l, x_r, x_r, x_l]
    y_tsg = [y_t, y_t, y_b, y_b]
    return x_tsg, y_tsg



def get_action_from_policy1d(policy_1d, state_tuple, g):
    s1_idx = s1_id(state_tuple)
    try:
        return g.actions[policy_1d[s1_idx][0]]
    except:
        print("ERROR")
        print('s1_idx= ', s1_idx)
        print("policy_1d[s1_idx][0] = ", policy_1d[s1_idx][0])
        


def plot_exact_trajectory_set_DP(g, policy_1d, X, Y, vel_field_data, scalar_field_data, fpath,
                                fname='Trajectories', 
                                with_policy_interp = False,
                                show_grid_policy = False, 
                                show_interp_policy_of_traj = False,
                                show_field_at_t_r = None,
                                show_scalar_field_at_t_r = None):

    # time calculation and state trajectory
    msize = 15
    # fsize = 3

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0,g.xs[-1] + (dx/2))
    ax.set_ylim(0,g.ys[-1] + (dy/2))

    minor_ticks = [i*g.dx/1 for i in range(gsize + 1)]
    major_ticks = [i*g.dx/1 for i in range(0, gsize + 1)]

    ax.set_xticks(minor_ticks, minor=True)
    ax.set_xticks(major_ticks, minor=False)
    ax.set_yticks(major_ticks, minor=False)
    ax.set_yticks(minor_ticks, minor=True)

    ax.grid(b= True, which='both', color='#CCCCCC', axis='both',linestyle = '-', alpha = 0.5)
    ax.tick_params(axis='both', which='both', labelsize=6)

    ax.set_xlabel('X (Non-Dim)')
    ax.set_ylabel('Y (Non-Dim)')

    st_point= g.start_state

    plt.scatter(g.xs[st_point[2]], g.ys[g.ni - 1 - st_point[1]], marker = 'o', s = msize, color = 'k', zorder = 1e5)
    x_ssg, y_ssg =  get_sg_square_corners(g, 'start')
    plt.fill(x_ssg, y_ssg, 'g', alpha = 0.5, zorder = 0)
    if (0<= g.endpos[0] < gsize) and (0<= g.endpos[1] < gsize):
        plt.scatter(g.xs[g.endpos[1]], g.ys[g.ni - 1 - g.endpos[0]], marker = '*', s = msize*2, color ='k', zorder = 1e5)
        x_tsg, y_tsg =  get_sg_square_corners(g, 'target')
        plt.fill(x_tsg, y_tsg, 'r', alpha = 0.5, zorder = 0)

    plt.gca().set_aspect('equal', adjustable='box')

    # plot obstacle
    obs_mask_mat = scalar_field_data[1]
    temp_t = 0
    for i in range(gsize):
        for j in range(gsize):
            if obs_mask_mat[temp_t,i,j] == 1:
                x_corners, y_corners = get_general_square_corners(g, (temp_t,i,j))
                plt.fill(x_corners, y_corners, 'k', alpha = 0.5, zorder = -1e2)

    # Also plots policy on grid
    if show_grid_policy == True:
        xtr=[]
        ytr=[]
        ax_list=[]
        ay_list=[]
        t = 0
        for i in range(gsize):
            for j in range(gsize):
                a = get_action_from_policy1d(policy_1d, (t,i,j), g)
                xtr.append(g.xs[j])
                ytr.append(g.ys[g.ni - 1 - i])
                a_x, a_y = action_to_quiver(a)
                ax_list.append(a_x)
                ay_list.append(a_y)

        plt.quiver(xtr, ytr, ax_list, ay_list, alpha = 0.5, color = 'b')

    # Also plots velocity field 
    if show_field_at_t_r != None:
        xtr = []
        ytr = []
        vx_list = []
        vy_list = []
        t,r = show_field_at_t_r
        for i in range(gsize):
            for j in range(gsize):
                xtr.append(g.xs[j])
                ytr.append(g.ys[g.ni - 1 - i])
                vx, vy = extract_velocity(vel_field_data, t, i, j, r)
                vx_list.append(vx)
                vy_list.append(vy)
        plt.quiver(xtr, ytr, vx_list, vy_list, color = 'c', alpha = 0.5)

    # Also plots scalar field
    if show_scalar_field_at_t_r != None:
        xtr=[]
        ytr=[]
        s_list = []
        all_s_mat = scalar_field_data[0]
        t,r = show_field_at_t_r
        for i in range(gsize):
            for j in range(gsize):
                xtr.append(g.xs[j])
                ytr.append(g.ys[g.ni - 1 - i])
                s = all_s_mat[t,i,j]
                s_list.append(s)
        s_arr = np.array(s_list)
        s_arr = s_arr.reshape((gsize, gsize))
        print("check shapes: ", X.shape, Y.shape, s_arr.shape)
        plt.contourf(X, Y, s_arr, cmap = "YlOrRd", alpha = 0.5, zorder = -1e5)



    bad_count =0
    traj_list = []
    t_list_all = []
    t_list_reached = []

    print("Trajectory plot: Starting rzn loop")
    for rzn in range(nrzns):
        print("rzn ", rzn)
        g.set_state(g.start_state)
        dont_plot =False
        bad_flag = False
        # t = 0
        G = 0

        xtr = []
        ytr = []
        ax_list = []
        ay_list = []

        t, i, j = g.start_state
        
        a = get_action_from_policy1d(policy_1d, g.current_state(), g)
        xtr.append(g.x)
        ytr.append(g.y)
        a_x, a_y = action_to_quiver(a)
        ax_list.append(a_x)
        ay_list.append(a_y)

        # while (not g.is_terminal()) and g.if_within_actionable_time():
        while True:
            vx, vy = extract_velocity(vel_field_data, t, i, j, rzn)
            # print(vx,"\t", vy,"\t",a)
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

            if (not g.is_terminal()) and  g.if_within_actionable_time():
                if with_policy_interp == True:
                    a = get_interpolated_action(policy_1d, g)
                else:
                    a = get_action_from_policy1d(policy_1d, g.current_state(), g)
                a_x, a_y = action_to_quiver(a)
                ax_list.append(a_x)
                ay_list.append(a_y)
                # print("i,j,a",i,j,a)


            elif g.is_terminal():
                break

            else:
            #  i.e. not terminal and not in actinable time.
            # already checked if ternminal or not. If not terminal 
            # if time reaches nt ie not within actionable time, then increment badcount and Dont plot
                bad_count+=1
                # bad_flag=True
                # dont_plot=True
                break

        # xy_traj_r = []  
        # for xy in zip(xtr, ytr):
        #     xy_traj_r.append(xy)
        # traj_list.append(xy_traj_r)
            
        if dont_plot==False:
            plt.plot(xtr, ytr)
            plt.scatter(xtr,ytr, s = 10)
            if show_interp_policy_of_traj:
                l = len(xtr)-1
                # plt.quiver(xtr[0:l], ytr[0:l], ax_list, ay_list, scale_units = 'x', scale =2)
                plt.quiver(xtr[0:l], ytr[0:l], ax_list, ay_list)

        if bad_flag==False:
            traj_list.append((xtr, ytr))
            t_list_all.append(t)
            t_list_reached.append(t)
        #ADDED for trajactory comparison
        else:
            traj_list.append(None)
            t_list_all.append(None)

    if fname != None:
        plt.savefig(join(fpath, fname),bbox_inches = "tight", dpi=300)
        # print("*** pickling traj_list ***")
        # picklePolicy(traj_list, join(fpath,fname))
        # print("*** pickled ***")

    bad_count_tuple = (bad_count, str(bad_count * 100 / nrzns) + '%')
    return t_list_all, t_list_reached, bad_count_tuple, traj_list




def plot_learned_policy(g, policy_1d, vel_field_data, scalar_field_data,
                         fpath, fname='Trajectories',
                         check_interp_at_intermed_points = False, 
                         show_field_at_t_r = None,
                         show_scalar_field_at_t_r = None):
    """
    Plots learned policy
    :param g: grid object
    :param show_field_at_t_r: None or tuple (t,r) 
    :param show_scalar_field_at_t_r: None or tuple (t,r_sc); r_sc is rzn for scalar field
    :param QL_params: [policy, Q, init_Q, label_data, filepath]  - details mentioned below
    :param showfig: whether you want to see fig during execution
    :return:
    """
    print("plot_policy: making policy plot")
    msize = 15

    # set grid

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0,g.xs[-1] + (dx/2))
    ax.set_ylim(0,g.ys[-1] + (dy/2))

    minor_ticks = [i*g.dx/1 for i in range(gsize + 1)]
    major_ticks = [i*g.dx/1 for i in range(0, gsize + 1)]

    ax.set_xticks(minor_ticks, minor=True)
    ax.set_xticks(major_ticks, minor=False)
    ax.set_yticks(major_ticks, minor=False)
    ax.set_yticks(minor_ticks, minor=True)

    ax.grid(b= True, which='both', color='#CCCCCC', axis='both',linestyle = '-', alpha = 0.5)
    ax.tick_params(axis='both', which='both', labelsize=6)

    ax.set_xlabel('X (Non-Dim)')
    ax.set_ylabel('Y (Non-Dim)')

    st_point= g.start_state
    plt.scatter(g.xs[st_point[2]], g.ys[g.ni - 1 - st_point[1]], marker = 'o', s = msize, color = 'k', zorder = 1e5)
    x_ssg, y_ssg =  get_sg_square_corners(g, 'start')
    plt.fill(x_ssg, y_ssg, 'g', alpha = 0.5, zorder = 0)
    if (0<= g.endpos[0] < gsize) and (0<= g.endpos[1] < gsize):
        plt.scatter(g.xs[g.endpos[1]], g.ys[g.ni - 1 - g.endpos[0]], marker = '*', s = msize*2, color ='k', zorder = 1e5)
        x_tsg, y_tsg =  get_sg_square_corners(g, 'target')
        plt.fill(x_tsg, y_tsg, 'r', alpha = 0.5, zorder = 0)
    plt.gca().set_aspect('equal', adjustable='box')

    xtr=[]
    ytr=[]
    ax_list=[]
    ay_list=[]
    vx_list=[]
    vy_list=[]

    t = 0
    for i in range(gsize):
        for j in range(gsize):
            a = get_action_from_policy1d(policy_1d, (t,i,j), g)
            xtr.append(g.xs[j])
            ytr.append(g.ys[g.ni - 1 - i])
            a_x, a_y = action_to_quiver(a)
            ax_list.append(a_x)
            ay_list.append(a_y)

    # plt.quiver(xtr, ytr, ax_list, ay_list, scale_units = 'x', scale =1)
    plt.quiver(xtr, ytr, ax_list, ay_list)
    


    if show_field_at_t_r != None:
        t,r = show_field_at_t_r
        for i in range(gsize):
            for j in range(gsize):
                vx, vy = extract_velocity(vel_field_data, t, i, j, r)
                vx_list.append(vx)
                vy_list.append(vy)
        plt.quiver(xtr, ytr, vx_list, vy_list, color = 'c', alpha = 0.5)
        # plt.quiver(xtr, ytr, vx_list, vy_list, color = 'c', alpha = 0.5, scale_units = 'x', scale =1)

    if show_scalar_field_at_t_r != None:
        s_list = []
        all_s_mat = scalar_field_data[0]
        t,r = show_field_at_t_r
        for i in range(gsize):
            for j in range(gsize):
                s = all_s_mat[t,i,j]
                s_list.append(s)
        s_arr = np.array(s_list)
        s_arr = s_arr.reshape((gsize, gsize))
        print("check shapes: ", X.shape, Y.shape, s_arr.shape)
        plt.contourf(X, Y, s_arr, cmap = "YlOrRd", alpha = 0.5, zorder = -1e5)
        # plt.quiver(xtr, ytr, vx_list, vy_list, color = 'c', alpha = 0.5, scale_units = 'x', scale =1)
    

    if check_interp_at_intermed_points:
        print("plot_policy: making interpolated policy plot")
        t = 0
        x_list = []
        y_list = []
        ax_list = []
        ay_list = []
        eps = 1e-4
        for i in range(1, gsize):
            for j in range(gsize-1):
                x0, y0 = get_cellcenter_from_state((t,i,j), g)
                x1, y1 = x0,                  y0 + (g.dy/2) - eps
                x2, y2 = x0 + (g.dx/2) - eps, y0
                x3, y3 = x0 + (g.dx/2) - eps, y0 + (g.dy/2) - eps
                point_list = [(x1, y1), (x2, y2), (x3, y3)]
                for xy in point_list:
                    x, y = xy
                    g.set_state((t,i,j), xcoord=x, ycoord=y)
                    if not g.is_terminal():
                        # print("(i,j) = ",i, ", ",j)
                        a = get_interpolated_action(policy_1d, g)
                        a_x, a_y = action_to_quiver(a)
                        # print("a_x, a_y = ",a_x, ", ", a_y)
                        # print()
                        ax_list.append(a_x)
                        ay_list.append(a_y)
                        x_list.append(x)
                        y_list.append(y)

        # print("check ax_list, ay_lsit")
        # print("lens", len(ax_list), len(ay_list), len(x_list), len(y_list))
        # for i in range(len(ax_list)): 
        #     try:
        #         print(i, ax_list[i][0],"\t", ay_list[i][0])
        #     except:
        #         print(i, ax_list[i], ay_list[i])
        # plt.quiver(x_list, y_list, ax_list, ay_list, scale_units = 'x', scale =1, alpha = 0.5, color ='b') 
        plt.quiver(x_list, y_list, ax_list, ay_list, alpha = 0.5, color ='b') 

    if fname != None:
        plt.savefig(join(fpath, fname),bbox_inches = "tight", dpi=600)


def get_refd_startpos_list(startpos, tsgsize):
    refd_startpos_list = []
    i0, j0 = startpos
    for i in range(tsgsize):
        for j in range(tsgsize):
            refd_startpos_list.append((i0 + i, j0 + j))

    return refd_startpos_list




if __name__ == "__main__":

    # prob_name = "all_jet_g10x10x10_r5"

    # file contains path to where modelOutput to be converted is stored
    file = open(r"temp_modelOp_dirName.txt","r") 

    # read lines as string from file line by line
    file_lines = file.readlines()
    prob_type = file_lines[0][0:-1]
    prob_name = file_lines[1][0:-1]
    prob_specs = file_lines[2][0:-1]
    modelOutput_path = join("src/", file_lines[3])
    print("read line:\n", 
            prob_type,"\n", 
            prob_name, "\n", 
            prob_specs, "\n", 
            modelOutput_path)
    
    # parameters
    params = np.load(join( join(ROOT_DIR, modelOutput_path), "prob_params.npy"))
    params = params.reshape((len(params),))
    gsize = int(params[0])
    dx = params[16]
    dy = params[17]
    tsgsize = int(params[12])
    nt = int(params[10])
    nrzns = int(params[2])
    num_ac_speeds = int(params[14])
    num_ac_angles = int(params[15])
    dt = params[4]
    F = params[3]
    endpos = (int(params[8]),int(params[9])) 
    startpos = (8,1) # (8,14) #

    print("CHECK PARAMS")
    print("gsize =", gsize)
    print("num_ac_speeds= ", num_ac_speeds)
    print("num_ac_angles= ", num_ac_angles)
    print("F= ",F)
    print("tsgsize= ", tsgsize)
    print("nt= ", nt)
    print("dx= ", dx)
    print("dy= ", dy)
    print("dt =", dt)
    print("nrzns= ", nrzns)
    print("endpos= ", endpos)
    print("startpos= ", startpos)

    # gsize = 10
    # dx = 1
    # dy = 1
    # tsgsize = 1
    # nt = 10
    # nrzns = 5
    # num_ac_speeds = 1
    # num_ac_angles = 8
    # dt = 1
    # F = 1
    # startpos = (7,1) #(14,2) #
    # endpos = (4,7) # (8,14) #
    # startpos = (160,20) #(80,10)
    # endpos = (100,180) #(50,90)

    num_actions = num_ac_speeds * num_ac_angles

    # get list of start cells, based on grid refinement defined by tsgsize
    refd_startpos_list = get_refd_startpos_list(startpos, tsgsize)

    # setup grid
    return_list = setup_grid(prob_name, num_ac_speeds = num_ac_speeds, num_ac_angles = num_ac_angles, nt = nt, dt =dt, F =F, 
                            startpos = startpos, endpos = endpos, Test_grid= True, gsize = gsize, dx = dx, dy = dy, tsgsize=tsgsize)
    g, xs, ys, X, Y, vel_field_data, nmodes, useful_num_rzns, paths, params, param_strsetup_grid, scalar_field_data = return_list

    # read policy
    fpath = join(ROOT_DIR, "src/data_solverOutput/" + prob_type + "/" + prob_name + "/" + prob_specs + "/")
    policy_1d =  np.load(join(fpath, "policy.npy"))
    value_fn_1d = np.load(join(fpath, "value_function.npy"))

    print("")
    print("fpath= ", fpath)
    print("len(policy) = ", len(policy_1d))

    plot_learned_policy(g, policy_1d, vel_field_data, scalar_field_data,
                         fpath, fname='Policy', 
                         check_interp_at_intermed_points= True, 
                         show_field_at_t_r = (0,0),
                         show_scalar_field_at_t_r = (0,0))

    t_sp_list = []
    interp_t_sp_list = []

    for i in range(len(refd_startpos_list)):

        startpos = refd_startpos_list[i]
        return_list = setup_grid(prob_name, num_ac_speeds = num_ac_speeds, num_ac_angles = num_ac_angles, nt = nt, dt =dt, F =F, 
                                startpos = startpos, endpos = endpos, Test_grid= True, gsize = gsize, dx = dx, dy = dy, tsgsize=tsgsize)
        g, xs, ys, X, Y, vel_field_data, nmodes, useful_num_rzns, paths, params, param_strsetup_grid, scalar_field_data = return_list

        t_data = plot_exact_trajectory_set_DP(g, policy_1d, X, Y, 
                                    vel_field_data, scalar_field_data,
                                    fpath,
                                    with_policy_interp = False,
                                    fname='Trajectories_sp'+ str(i), 
                                    show_grid_policy = False, 
                                    show_interp_policy_of_traj = True,
                                    show_field_at_t_r=(0,0),
                                    show_scalar_field_at_t_r=(0,0))

        interp_t_data = plot_exact_trajectory_set_DP(g, policy_1d, X, Y, 
                                    vel_field_data, scalar_field_data,
                                    fpath,
                                    with_policy_interp = True,
                                    fname='Trajectories_interp_pol_sp'+ str(i), 
                                    show_grid_policy = False, 
                                    show_interp_policy_of_traj = True,
                                    show_field_at_t_r=(0,0),
                                    show_scalar_field_at_t_r=(0,0))

        t_list_all, t_list_reached, bad_count_tuple, traj_list = t_data
        t_sp_list.append(t_list_reached[0])
        np.save(join(fpath,'traj_list'), traj_list)
        # print("NO Interp: first succ EAT = ", t_list_reached[0])
        t_list_all, t_list_reached, bad_count_tuple, traj_list = interp_t_data
        interp_t_sp_list.append(t_list_reached[0])
        np.save(join(fpath,'traj_list_wip'), traj_list)

        # print("With Interp: first succ EAT = ", t_list_reached[0])

    print("No interpol: ", t_sp_list, "\navg=", np.mean(t_sp_list))
    print("WITH interpol: ", interp_t_sp_list, "\navg=", np.mean(interp_t_sp_list))
    # verbose_compare_trajs(traj1, traj2)

    # print(policy[])

# for t in range(nt):
#     for i in range(gsize):
#         for j in range(gsize):
#             # print(get_action_from_policy1d(policy_1d, (t,i,j), g) , end ='')
#             print(policy_1d[s1_id((t,i,j))][0] , end =' ')

#         print()
#     print("\n\n")


# for t in range(nt):
#     for i in range(gsize):
#         for j in range(gsize):
#             # print(get_action_from_policy1d(policy_1d, (t,i,j), g) , end ='')
#             print(value_fn_1d[s1_id((t,i,j))][0] , end =' ')

#         print()
    # print("\n\n")