import numpy as np
from utils.setup_grid import setup_grid
from definition import ROOT_DIR
from os.path import join
from utils.custom_functions import extract_velocity, get_angle_in_0_2pi
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import math
import imageio
import cv2
from pathlib import Path
import csv
import time
import pandas

g_strmplot_arrowsize  = 3.0 # scaling factor
g_strmplot_lw = 2 # linewidth of streamplot

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
        


def setup_grid_in_plot(fig, ax, g):

    msize = 100
    ax.set_xlim(0,g.xs[-1] + (dx/2))
    ax.set_ylim(0,g.ys[-1] + (dy/2))

    minor_ticks = [i*g.dx/1 for i in range(0, gsize + 1, 10)]
    major_ticks = [i*g.dx/1 for i in range(0, gsize + 1, 40)]

    ax.set_xticks(minor_ticks, minor=True)
    ax.set_xticks(major_ticks, minor=False)
    ax.set_yticks(major_ticks, minor=False)
    ax.set_yticks(minor_ticks, minor=True)

    # ax.grid(b= True, which='both', color='#CCCCCC', axis='both',linestyle = '-', alpha = 0.5, zorder = -1e5)
    ax.tick_params(axis='both', which='both', labelsize=20)

    ax.set_xlabel('X (Non-Dim)', fontsize = 30)
    ax.set_ylabel('Y (Non-Dim)', fontsize = 30)

    st_point= g.start_state

    plt.scatter(g.xs[st_point[2]], g.ys[g.ni - 1 - st_point[1]], marker = 'o', s = msize, color = 'k', zorder = 1e5)
    x_ssg, y_ssg =  get_sg_square_corners(g, 'start')
    plt.fill(x_ssg, y_ssg, 'g', alpha = 0.5, zorder = 0)
    if (0<= g.endpos[0] < gsize) and (0<= g.endpos[1] < gsize):
        plt.scatter(g.xs[g.endpos[1]], g.ys[g.ni - 1 - g.endpos[0]], marker = '*', s = msize*2, color ='k', zorder = 1e5)
        x_tsg, y_tsg =  get_sg_square_corners(g, 'target')
        plt.fill(x_tsg, y_tsg, 'r', alpha = 0.5, zorder = 0)

    plt.gca().set_aspect('equal', adjustable='box')

    return

def get_xy_from_ij(i,j,g):
    x = g.xs[j]
    y = g.ys[g.ni - 1 - i]
    return x,y

def get_xy_list(g, gsize):
    x_list=[]
    y_list=[]
    for i in range(gsize):
        for j in range(gsize):
            x, y = get_xy_from_ij(i,j,g)
            x_list.append(x)
            y_list.append(y)
    return (x_list, y_list)


def func_show_grid_policy(t, policy_1d, g, gsize, xy_list=None):
    if xy_list==None:
        x_list=[]
        y_list=[]
    else:
        x_list, y_list = xy_list
    ax_list=[]
    ay_list=[]
    for i in range(gsize):
        for j in range(gsize):
            a = get_action_from_policy1d(policy_1d, (t,i,j), g)
            if xy_list==None:
                x, y = get_xy_from_ij(i,j,g)
                x_list.append(x)
                y_list.append(y)
            a_x, a_y = action_to_quiver(a)
            ax_list.append(a_x)
            ay_list.append(a_y)
    return x_list, y_list, ax_list, ay_list


def func_show_velocity_field(t_r, vel_field_data, g, gsize, xy_list=None):
    if xy_list==None:
        x_list=[]
        y_list=[]
    else:
        x_list, y_list = xy_list
    vx_list = []
    vy_list = []
    t, r = t_r
    for i in range(gsize):
        for j in range(gsize):
            if xy_list == None:
                x, y = get_xy_from_ij(i,j,g)
                x_list.append(x)
                y_list.append(y)
            vx, vy = extract_velocity(vel_field_data, t, i, j, r)
            vx_list.append(vx)
            vy_list.append(vy)
    return x_list, y_list, vx_list, vy_list


def func_show_scalar_field(t_r, scalar_field_data, g, gsize, xy_list=None):
    if xy_list==None:
        x_list=[]
        y_list=[]
    else:
        x_list, y_list = xy_list
    s_list = []
    all_s_mat = scalar_field_data[0]
    t,_ = t_r
    for i in range(gsize):
        for j in range(gsize):
            if xy_list == None:
                x, y = get_xy_from_ij(i,j,g)
                x_list.append(x)
                y_list.append(y)
            s = all_s_mat[t,i,j]
            s_list.append(s)
    s_arr = np.array(s_list)
    s_arr = s_arr.reshape((gsize, gsize))
    # print("check shapes: ", X.shape, Y.shape, s_arr.shape)
    return s_arr

def func_show_obstacles(t, obs_mask_mat, g, gsize):
    for i in range(gsize):
        for j in range(gsize):
            if obs_mask_mat[t,i,j] == 1:
                x_corners, y_corners = get_general_square_corners(g, (t,i,j))
                plt.fill(x_corners, y_corners, color ='dimgray', alpha = 1, zorder = 1)
    return


def in_obstacle(obstacle_mask, t, i, j):
    if obstacle_mask[t,i,j]==1:
        return True
    else:
        return False

def plot_exact_trajectory_set_DP(g, policy_1d, X, Y, vel_field_data, scalar_field_data,
                                nrzns, nrzns_to_plot, fpath,
                                fname='Trajectories', 
                                with_policy_interp = False,
                                show_grid_policy = False, 
                                show_interp_policy_of_traj = False,
                                show_field_at_t_r = None,
                                show_scalar_field_at_t_r = None):

    # time calculation and state trajectory
    # fsize = 3
    traj_list = []
    Cf = 1
    Cr = 1

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    setup_grid_in_plot(fig, ax, g)

    # plot obstacle
    obs_mask_mat = scalar_field_data[1]
    t = 0
    func_show_obstacles(t, obs_mask_mat, g, gsize)
 
    # get x_list and y_list of cells in grid
    xy_list = get_xy_list(g, gsize)
    x_list, y_list = xy_list

    # Also plots policy on grid
    if show_grid_policy == True:
        t=0
        _,_,ax_list, ay_list = func_show_grid_policy(t, policy_1d, g, gsize, xy_list=xy_list)
        plt.quiver(x_list, y_list, ax_list, ay_list, alpha = 0.5, color = 'b')

    # Also plots velocity field 
    if show_field_at_t_r != None:
        t_r = show_field_at_t_r
        _,_,vx_list, vy_list = func_show_velocity_field(t_r, vel_field_data, g, gsize, xy_list=xy_list)
        vx_grid = np.reshape(np.array(vx_list), (gsize,gsize))
        vy_grid = np.reshape(np.array(vy_list), (gsize,gsize))
        plt.streamplot(X, Y, vx_grid, vy_grid, color = 'k', zorder = 0,  linewidth=g_strmplot_lw, arrowsize=g_strmplot_arrowsize, arrowstyle='->')
        # plt.quiver(x_list, y_list, vx_list, vy_list, color = 'c', alpha = 0.5)


    # Also plots scalar field
    if show_scalar_field_at_t_r != None:
        t_r = 0,0
        s_arr = func_show_scalar_field(t_r, scalar_field_data, g, gsize, xy_list=xy_list)
        plt.contourf(X, Y, s_arr, cmap = "YlOrRd_r", alpha = 0.5, zorder = -1e5)



    print("Trajectory plot: Starting rzn loop")
    for rzn in range(nrzns):
        success = None
        energy_cons = 0
        energy_col = 0
        net_energy_cons = 0
        travel_time = 0

        if rzn%100 == 0 or rzn ==  nrzns-1:
            print("rzn ", rzn)
        g.set_state(g.start_state)

        # t = 0
        G = 0

        xtr = []
        ytr = []
        ax_list = []
        ay_list = []

        t, i, j = g.start_state
        assert(t==0)
        
        a = get_action_from_policy1d(policy_1d, g.current_state(), g)
        xtr.append(g.x)
        ytr.append(g.y)
        a_x, a_y = action_to_quiver(a)
        ax_list.append(a_x)
        ay_list.append(a_y)

        # while (not g.is_terminal()) and g.if_within_actionable_time():
        for iter in range(g.nt):

            if success == None:
                vx, vy = extract_velocity(vel_field_data, t, i, j, rzn)
                # print(vx,"\t", vy,"\t",a)
                rad1 = scalar_field_data[0][t,i,j]
                r = g.move_exact(a, vx, vy)
                G = G + r
                # t += 1
                (t, i, j) = g.current_state()
                rad2 = scalar_field_data[0][t,i,j]
                travel_time += 1
                energy_cons += (Cf*a[0])**2
                energy_col += (Cr*(rad2 + rad1)/2)
                net_energy_cons += (Cf*(a[0]**2)) - (Cr*(rad2 + rad1)/2) 

            xtr.append(g.x)
            ytr.append(g.y)


            # if edge state encountered, then increment badcount and Dont plot
            if g.if_edge_state((i,j)) or in_obstacle(scalar_field_data[1], t, i ,j):
                success = False

            if (not g.is_terminal()) and  g.if_within_actionable_time() and success == None:
                if with_policy_interp == True:
                    a = get_interpolated_action(policy_1d, g)
                else:
                    a = get_action_from_policy1d(policy_1d, g.current_state(), g)
                a_x, a_y = action_to_quiver(a)
                ax_list.append(a_x)
                ay_list.append(a_y)
                # print("i,j,a",i,j,a)

            elif g.is_terminal():
                success = True
  
            else:
            #  i.e. not terminal and not in actinable time.
            # already checked if ternminal or not. If not terminal 
            # if time reaches nt ie not within actionable time, 
                success = False

        plt.plot(xtr, ytr)
        plt.scatter(xtr,ytr, s = 10)
        if show_interp_policy_of_traj:
            l = travel_time
            # plt.quiver(xtr[0:l], ytr[0:l], ax_list, ay_list, scale_units = 'x', scale =2)
            plt.quiver(xtr[0:l], ytr[0:l], ax_list, ay_list)

        traj_list.append((xtr, ytr, success, travel_time, energy_cons, energy_col, net_energy_cons))


    if fname != None:
        plt.savefig(join(fpath, fname),bbox_inches = "tight", dpi=300)


    return traj_list



def func_check_interp_at_intermediate_points(t, policy_1d, g, gsize):
        print("plot_policy: making interpolated policy plot")
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
        return x_list, y_list, ax_list, ay_list


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
    # set grid
    # obs_mask_mat = scalar_field_data[1]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    setup_grid_in_plot(fig, ax, g)

    xy_list = get_xy_list(g, gsize)
    x_list, y_list = xy_list
  
    t=0
    _,_,ax_list, ay_list = func_show_grid_policy(t, policy_1d, g, gsize, xy_list=xy_list)
    plt.quiver(x_list, y_list, ax_list, ay_list)
    
    if show_field_at_t_r != None:
        t_r = show_field_at_t_r
        _,_,vx_list, vy_list = func_show_velocity_field(t_r, vel_field_data, g, gsize, xy_list=xy_list)
        plt.quiver(x_list, y_list, vx_list, vy_list, color = 'c', alpha = 0.5)
        # plt.quiver(xtr, ytr, vx_list, vy_list, color = 'c', alpha = 0.5, scale_units = 'x', scale =1)

    if show_scalar_field_at_t_r != None:
        t_r = 0,0
        s_arr = func_show_scalar_field(t_r, scalar_field_data, g, gsize, xy_list=xy_list)
        plt.contourf(X, Y, s_arr, cmap = "YlOrRd_r", alpha = 0.5, zorder = -1e5)
        # plt.quiver(xtr, ytr, vx_list, vy_list, color = 'c', alpha = 0.5, scale_units = 'x', scale =1)
    

    if check_interp_at_intermed_points:
        func_check_interp_at_intermediate_points(t, policy_1d, g, gsize)
        # plt.quiver(x_list, y_list, ax_list, ay_list, scale_units = 'x', scale =1, alpha = 0.5, color ='b') 
        plt.quiver(x_list, y_list, ax_list, ay_list, alpha = 0.5, color ='b') 

    if fname != None:
        plt.savefig(join(fpath, fname),bbox_inches = "tight", dpi=600)

    return



# Use this function for obtaining time only (NO PLOTS GENERATED)
def plot_exact_trajectory_set_DP_dummy(g, policy_1d, X, Y, vel_field_data, scalar_field_data,
                                nrzns, nrzns_to_plot, fpath,
                                fname='Trajectories', 
                                with_policy_interp = False,
                                show_grid_policy = False, 
                                show_interp_policy_of_traj = False,
                                show_field_at_t_r = None,
                                show_scalar_field_at_t_r = None):

    # time calculation and state trajectory
    # fsize = 3
    traj_list = []
    Cf = 1
    Cr = 1

    #fig = plt.figure(figsize=(10, 10))
    #ax = fig.add_subplot(1, 1, 1)
    #setup_grid_in_plot(fig, ax, g)

    # plot obstacle
    obs_mask_mat = scalar_field_data[1]
    t = 0
    #func_show_obstacles(t, obs_mask_mat, g, gsize)
 
    # get x_list and y_list of cells in grid
    xy_list = get_xy_list(g, gsize)
    x_list, y_list = xy_list

    # Also plots policy on grid
    #if show_grid_policy == True:
    #    t=0
    #    _,_,ax_list, ay_list = func_show_grid_policy(t, policy_1d, g, gsize, xy_list=xy_list)
    #    plt.quiver(x_list, y_list, ax_list, ay_list, alpha = 0.5, color = 'b')

    # Also plots velocity field 
    #if show_field_at_t_r != None:
    #    t_r = show_field_at_t_r
    #    _,_,vx_list, vy_list = func_show_velocity_field(t_r, vel_field_data, g, gsize, xy_list=xy_list)
    #    vx_grid = np.reshape(np.array(vx_list), (gsize,gsize))
    #    vy_grid = np.reshape(np.array(vy_list), (gsize,gsize))
    #    plt.streamplot(X, Y, vx_grid, vy_grid, color = 'k', zorder = 0,  linewidth=g_strmplot_lw, arrowsize=g_strmplot_arrowsize, arrowstyle='->')
    #    # plt.quiver(x_list, y_list, vx_list, vy_list, color = 'c', alpha = 0.5)


    # Also plots scalar field
    #if show_scalar_field_at_t_r != None:
    #    t_r = 0,0
    #    s_arr = func_show_scalar_field(t_r, scalar_field_data, g, gsize, xy_list=xy_list)
    #    plt.contourf(X, Y, s_arr, cmap = "YlOrRd_r", alpha = 0.5, zorder = -1e5)



    print("Trajectory plot: Starting rzn loop")
    for rzn in range(nrzns):
        success = None
        energy_cons = 0
        energy_col = 0
        net_energy_cons = 0
        travel_time = 0

        if rzn%100 == 0 or rzn ==  nrzns-1:
            print("rzn ", rzn)
        g.set_state(g.start_state)

        # t = 0
        G = 0

        xtr = []
        ytr = []
        ax_list = []
        ay_list = []

        t, i, j = g.start_state
        assert(t==0)
        
        a = get_action_from_policy1d(policy_1d, g.current_state(), g)
        xtr.append(g.x)
        ytr.append(g.y)
        a_x, a_y = action_to_quiver(a)
        ax_list.append(a_x)
        ay_list.append(a_y)

        # while (not g.is_terminal()) and g.if_within_actionable_time():
        for iter in range(g.nt):

            if success == None:
                vx, vy = extract_velocity(vel_field_data, t, i, j, rzn)
                # print(vx,"\t", vy,"\t",a)
                rad1 = scalar_field_data[0][t,i,j]
                r = g.move_exact(a, vx, vy)
                G = G + r
                # t += 1
                (t, i, j) = g.current_state()
                rad2 = scalar_field_data[0][t,i,j]
                travel_time += 1
                energy_cons += (Cf*a[0])**2
                energy_col += (Cr*(rad2 + rad1)/2)
                net_energy_cons += (Cf*(a[0]**2)) - (Cr*(rad2 + rad1)/2) 

            xtr.append(g.x)
            ytr.append(g.y)


            # if edge state encountered, then increment badcount and Dont plot
            if g.if_edge_state((i,j)) or in_obstacle(scalar_field_data[1], t, i ,j):
                success = False

            if (not g.is_terminal()) and  g.if_within_actionable_time() and success == None:
                if with_policy_interp == True:
                    a = get_interpolated_action(policy_1d, g)
                else:
                    a = get_action_from_policy1d(policy_1d, g.current_state(), g)
                a_x, a_y = action_to_quiver(a)
                ax_list.append(a_x)
                ay_list.append(a_y)
                # print("i,j,a",i,j,a)

            elif g.is_terminal():
                success = True
  
            else:
            #  i.e. not terminal and not in actinable time.
            # already checked if ternminal or not. If not terminal 
            # if time reaches nt ie not within actionable time, 
                success = False

        #plt.plot(xtr, ytr)
        #plt.scatter(xtr,ytr, s = 10)
        if show_interp_policy_of_traj:
            l = travel_time
            # plt.quiver(xtr[0:l], ytr[0:l], ax_list, ay_list, scale_units = 'x', scale =2)
            #plt.quiver(xtr[0:l], ytr[0:l], ax_list, ay_list)

        traj_list.append((xtr, ytr, success, travel_time, energy_cons, energy_col, net_energy_cons))


    #if fname != None:
    #    plt.savefig(join(fpath, fname),bbox_inches = "tight", dpi=300)


    return traj_list


def dynamic_plot_sequence_and_gif(traj_list, g, policy_1d, 
                            vel_field_data, scalar_field_data,nrzns, nrzns_to_plot,
                            fpath, fname='Trajectories'):


    xy_list = get_xy_list(g, gsize)
    x_list, y_list = xy_list
    len_list = [len(traj[0]) for traj in traj_list]
    # print("------CHECK----")
    # print(np.min(len_list),np.max(len_list))
    # print(len_list[0:10])
    rzn_list = [i for i in range(nrzns)]
    images = []
    for t in range(nt):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        setup_grid_in_plot(fig, ax, g)
        title = "t = " + str(t)
        plt.title(title)

        obs_mask_mat = scalar_field_data[1]
        func_show_obstacles(t, obs_mask_mat, g, gsize)    

        _,_,vx_list, vy_list = func_show_velocity_field((t,0), vel_field_data, g, gsize, xy_list=xy_list)
        vx_grid = np.reshape(np.array(vx_list), (gsize,gsize))
        vy_grid = np.reshape(np.array(vy_list), (gsize,gsize))
        plt.streamplot(X, Y, vx_grid, vy_grid, color = 'k', zorder = -1e4,  linewidth=g_strmplot_lw, arrowsize=g_strmplot_arrowsize, arrowstyle='->')
        # plt.quiver(x_list, y_list, vx_list, vy_list, color = 'c', alpha = 0.5)

        s_arr = func_show_scalar_field((t,0), scalar_field_data, g, gsize, xy_list=xy_list)
        plt.contourf(X, Y, s_arr, cmap = "YlOrRd_r", alpha = 0.5, zorder = -1e5)
        plt.colorbar()

        # jet = cm = plt.get_cmap('jet')
        # cNorm = colors.Normalize(vmin=np.min(len_list), vmax=np.max(len_list))
        # scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        # scalarMap._A = []


        for k in rzn_list:
            xtr, ytr = traj_list[k]
            try:
                # colorval = scalarMap.to_rgba(len_list[k])
                plt.plot(xtr[0:t+1], ytr[0:t+1], zorder = 1e2)
                plt.scatter(xtr[0:t], ytr[0:t], s=10, zorder = 1e2)
                plt.scatter(xtr[t], ytr[t], color = 'k', marker = '^', s = 20, zorder = 1e5)
            except:
                pass

        filename = join(fpath, fname)
        plt.savefig(filename, dp = 50)
        plt.clf()
        plt.close()
        images.append(imageio.imread(filename+".png"))

        if t > np.max(len_list):
            "breaking"
            break
  
    gif_name = filename + ".gif"
    imageio.mimsave(gif_name, images, duration = 0.5)

    return
    

def dynamic_plot_sequence(traj_list, metric_data, g, policy_1d, 
                            vel_field_data, scalar_field_data,nrzns, nrzns_to_plot,
                            fpath, plot_interval, prob_type, plot_at_t =None, fname='Trajectories'):

                            
    # traj_list = [ (xtr, ytr, success, travel_time, energy_cons, net_energy_cons), (), ()... ]
    plot_seq_path = join(fpath, "plot_sequence")
    Path(plot_seq_path).mkdir(parents=False, exist_ok=True)
    xy_list = get_xy_list(g, gsize)
    x_list, y_list = xy_list
    len_list = [len(traj[0]) for traj in traj_list]
    rzn_list = [i for i in range(nrzns)]

    travel_time_list, energy_cons_list, energy_col_list, net_energy_cons_list, success_rate = metric_data

    cmap = plt.get_cmap('plasma')
    if prob_type == "time":
        vmin, vmax = np.min(travel_time_list), np.max(travel_time_list)
    if prob_type == "energy1":
        vmin, vmax = np.min(energy_cons_list), np.max(energy_cons_list)
    if prob_type == "energy2":
        vmin, vmax = np.min(net_energy_cons_list), np.max(net_energy_cons_list)
    if prob_type == "energy3":
        vmin, vmax = np.min(energy_col_list), np.max(energy_col_list)
    if prob_type == "custom1":
        vmin, vmax = np.min(energy_cons_list), np.max(energy_cons_list)
    if prob_type == "custom2":
        vmin, vmax = np.min(net_energy_cons_list), np.max(net_energy_cons_list)
    if prob_type == "custom3":
        vmin, vmax = np.min(energy_col_list), np.max(energy_col_list)
    if prob_type == "custom4":
        vmin, vmax = np.min(energy_cons_list), np.max(energy_cons_list)

    vmin = 40
    vmax = 200
    cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    scalarMap._A = []

    for t in range(g.nt):
        #### this has been capped for t = 60 units
        # (((plot_at_t==None) and (t%plot_interval == 0 or t==1 or t == np.max(metric_data[0]) or t == nt-1)) or ((plot_at_t != None) and (t in plot_at_t or t==1)) and t < 60)
        if (((plot_at_t==None) and (t%plot_interval == 0 or t==1 or t == np.max(metric_data[0]) or t == nt-1)) or ((plot_at_t != None) and (t in plot_at_t or t==1)) and t < 60):

            print("t=",t)
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(1, 1, 1)
            setup_grid_in_plot(fig, ax, g)
            title = "t = " + str(t)
            plt.title(title, fontsize=40)

            xt = np.array([0,25,50,75,100])
            plt.xticks(xt, fontsize=30)
            plt.yticks(xt,fontsize=30)

            obs_mask_mat = scalar_field_data[1]
            func_show_obstacles(t, obs_mask_mat, g, gsize)    

            _,_,vx_list, vy_list = func_show_velocity_field((t,0), vel_field_data, g, gsize, xy_list=xy_list)
            vx_grid = np.reshape(np.array(vx_list), (gsize,gsize))
            vy_grid = np.reshape(np.array(vy_list), (gsize,gsize))
            plt.streamplot(X, Y, vx_grid, vy_grid, color = 'grey', zorder = -1e4, linewidth=g_strmplot_lw, arrowsize=g_strmplot_arrowsize, arrowstyle='->')
            #plt.quiver(x_list, y_list, vx_list, vy_list, color = 'c', alpha = 0.5)

            s_arr = func_show_scalar_field((t,0), scalar_field_data, g, gsize, xy_list=xy_list)
            if prob_type == str('energy1') or prob_type == str('time') or prob_type == str('custom1') or prob_type == str('custom4'):
                vel_mag = (vx_grid**2 + vy_grid**2)**0.5
                print(np.amax(vel_mag))
                plt.contourf(X, Y, vel_mag, cmap = "Blues", zorder = -1e5)
            else:
                plt.contourf(X, Y, s_arr, cmap = "YlOrRd_r", alpha = 0.45, zorder = -1e5)
            plt.colorbar()

            for k in rzn_list:
                # print("rzn",k)

                xtr, ytr, success, travel_time, energy_cons, energy_col, net_energy_cons = traj_list[k]
                # print("check xtr", xtr)
                if success == True:
                    if prob_type == "time":
                        colorval = scalarMap.to_rgba(travel_time)
                    if prob_type == "energy1":
                        colorval = scalarMap.to_rgba(energy_cons)
                    if prob_type == "energy2":
                        colorval = scalarMap.to_rgba(net_energy_cons)
                    if prob_type == "energy3":
                        colorval = scalarMap.to_rgba(energy_col)
                    if prob_type == "custom1":
                        colorval = scalarMap.to_rgba(energy_cons)
                    if prob_type == "custom2":
                        colorval = scalarMap.to_rgba(net_energy_cons)
                    if prob_type == "custom3":
                        colorval = scalarMap.to_rgba(energy_col)
                    if prob_type == "custom4":
                        colorval = scalarMap.to_rgba(energy_cons)                        
                    try:
                        # colorval = scalarMap.to_rgba(len_list[k])
                        plt.plot(xtr[0:t+1], ytr[0:t+1], linewidth=1, color=colorval, zorder = 1e4)
                        # plt.scatter(xtr[0:t], ytr[0:t], s=10, zorder = 1e2)
                        plt.scatter(xtr[t], ytr[t], color = 'k', marker = '^', s = 40, zorder = 1e5)
                    except:
                        pass
            if t == 1:
                cbar = plt.colorbar(scalarMap)
                cbar.ax.tick_params(labelsize=20) 

            filename = join(plot_seq_path, fname) + "_newT_@t" + str(t) + "_paper"+".png"
            plt.savefig(filename, bbox_inches = "tight", dp = 300)
            plt.clf()
            plt.close()

            if t > np.max(len_list):
                "breaking"
                break
  

    return

def get_refd_startpos_list(startpos, tsgsize):
    refd_startpos_list = []
    i0, j0 = startpos
    for i in range(tsgsize):
        for j in range(tsgsize):
            refd_startpos_list.append((i0 + i, j0 + j))

    return refd_startpos_list

def get_metrics(traj_data):
    # traj_data = (xtr, ytr, success, travel_time, energy_cons, net_energy_cons)
    sum_success = 0

    travel_time_list = []
    energy_cons_list = []
    energy_col_list = []
    net_energy_cons_list = []
    success_list = []
    for traj in traj_data:
        _, _, success, travel_time, energy_cons, energy_col, net_energy_cons = traj

        if success == True:
            sum_success += 1
            travel_time_list.append(travel_time)
            energy_cons_list.append(energy_cons)

            energy_col_list.append(energy_col)
            net_energy_cons_list.append(net_energy_cons)
    success_rate = sum_success/len(traj_data)

    return [travel_time_list, energy_cons_list, energy_col_list, net_energy_cons_list, success_rate]

def write_log(metric_data, prob_list, design_param, build_time, spvi_time, plot_time, data_time, build_time_only):

    # metric data [travel_time_list, energy_cons_list, energy_col_list, net_energy_cons_list, success_rate]

    success_rate = metric_data[4]
    time_min = np.min(metric_data[0])
    time_max = np.max(metric_data[0])
    time_mean = np.mean(metric_data[0])
    reqd_energy_min = np.min(metric_data[1])
    reqd_energy_max = np.max(metric_data[1])
    reqd_energy_mean = np.mean(metric_data[1])
    coll_energy_min = np.min(metric_data[2])
    coll_energy_max = np.max(metric_data[2])
    coll_energy_mean = np.mean(metric_data[2])
    net_energy_min = np.min(metric_data[3])
    net_energy_max = np.max(metric_data[3])
    net_energy_mean = np.mean(metric_data[3])

    list2 = [success_rate, time_min, time_max, time_mean, reqd_energy_min, reqd_energy_max, reqd_energy_mean, coll_energy_min, coll_energy_max, coll_energy_mean, net_energy_min, net_energy_max, net_energy_mean]
    time_list = [build_time, spvi_time, plot_time, data_time, build_time_only]
    row = prob_list + list2 + design_param + time_list

    with open('/gdata1/rohitc/e2e_GPU_DP/src/log.csv', 'a') as f_object:
        writer_object = csv.writer(f_object)
        writer_object.writerow(row)
        f_object.close()

    return





if __name__ == "__main__":

    start_time = time.time()

    # prob_name = "all_jet_g10x10x10_r5"

    # file contains path to where modelOutput to be converted is stored
    file = open(r"temp_modelOp_dirName.txt","r") 

    # read lines as string from file line by line
    file_lines = file.readlines()
    print(file_lines)
    prob_type = file_lines[0][0:-1]
    prob_name = file_lines[1][0:-1]
    prob_specs = file_lines[2][0:-1]
    alpha_str = file_lines[3][0:-1]
    modelOutput_path = join("src/", file_lines[4])
    print("read line:\n", 
            prob_type,"\n", 
            prob_name, "\n", 
            prob_specs, "\n",
            alpha_str, "\n", 
            modelOutput_path)
    print("prob_type=",prob_type)
    prob_type = str(prob_type)

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
    alpha = params[19]

    # startpos = (int(0.2*gsize), int(0.2*gsize))
    startpos = (int(0.85*gsize), int(0.4*gsize))
    nrzns_to_plot = 100

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



    num_actions = num_ac_speeds * num_ac_angles

    # get list of start cells, based on grid refinement defined by tsgsize
    refd_startpos_list = get_refd_startpos_list(startpos, tsgsize)

    # setup grid
    return_list = setup_grid(prob_name, num_ac_speeds = num_ac_speeds, num_ac_angles = num_ac_angles, nt = nt, dt =dt, F =F, 
                            startpos = startpos, endpos = endpos, Test_grid= True, gsize = gsize, dx = dx, dy = dy, tsgsize=tsgsize)
    g, xs, ys, X, Y, vel_field_data, nmodes, _, paths, params, param_strsetup_grid, scalar_field_data = return_list

    # read policy

    if prob_type == "custom1" or prob_type == "custom2" or prob_type == "custom3" or prob_type == "custom4":
        fpath = join(ROOT_DIR, "src/data_solverOutput/" + prob_type + "/" + prob_name + "/" + prob_specs + "/" + alpha_str + "/")
    else:
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

    traj_data_start_time = time.time()

    # for i in range(len(refd_startpos_list)):
    for i in range(1):

        startpos = refd_startpos_list[i]
        # return_list = setup_grid(prob_name, num_ac_speeds = num_ac_speeds, num_ac_angles = num_ac_angles, nt = nt, dt =dt, F =F, 
        #                         startpos = startpos, endpos = endpos, Test_grid= True, gsize = gsize, dx = dx, dy = dy, tsgsize=tsgsize)
        # g, xs, ys, X, Y, vel_field_data, nmodes, useful_num_rzns, paths, params, param_strsetup_grid, scalar_field_data = return_list

        # traj_data = plot_exact_trajectory_set_DP(g, policy_1d, X, Y, 
        #                             vel_field_data, scalar_field_data, nrzns, nrzns_to_plot,
        #                             fpath,
        #                             with_policy_interp = False,
        #                             fname='Trajectories_sp'+ str(i), 
        #                             show_grid_policy = False, 
        #                             show_interp_policy_of_traj = False,
        #                             show_field_at_t_r=(0,0),
        #                             show_scalar_field_at_t_r=(0,0))



        interp_traj_data = plot_exact_trajectory_set_DP(g, policy_1d, X, Y,            ############## CHANGED #######################
                                    vel_field_data, scalar_field_data, nrzns, nrzns_to_plot,
                                    fpath,
                                    with_policy_interp = True,
                                    fname='Trajectories_interp_pol_sp'+ str(i), 
                                    show_grid_policy = False, 
                                    show_interp_policy_of_traj = False,
                                    show_field_at_t_r=(0,0),
                                    show_scalar_field_at_t_r=(0,0))


        interp_traj_data_np = np.array(interp_traj_data, dtype=object)
        np.save(join(fpath,'interp_traj_data'), interp_traj_data_np, allow_pickle=True)  ################## CHNAGED #########################

        # print("NO Interp: first succ EAT = ", t_list_reached[0])
        # np.save(join(fpath,'interp_traj_data'), interp_traj_data)

        # metric_data = get_metrics(traj_data)
        interp_metric_data = get_metrics(interp_traj_data)

    traj_data_end_time = time.time()

    prob_list = [prob_type, prob_name, prob_specs, startpos[0], startpos[1]]
    design_param = [alpha]


    # write_log(interp_metric_data, prob_list, design_param)

    try:
        print("No interpol: ", )
        # print("mean_travel_time, mean_en_cons, mean_net_en_cons, success_rate= ", metrics)
        print("Success rate= ", metric_data[3])
        print("\t ---")
        print("Travel Time:")
        print(" min:", np.min(metric_data[0]))
        print(" max:", np.max(metric_data[0]))
        print(" mean:", np.mean(metric_data[0]))
        print("\t ---")
        if prob_type == "energy1":
            print("Energy Consumed:")
            print(" min:", np.min(metric_data[1]))
            print(" max:", np.max(metric_data[1]))
            print(" mean:", np.mean(metric_data[1]))
            print("\t ---")
        if prob_type == "energy2":
            print("Net Energy Consumed:")
            print(" min:", np.min(metric_data[2]))
            print(" max:", np.max(metric_data[2]))
            print(" mean:", np.mean(metric_data[2]))
            print("\t ---")
        if prob_type == "custom1":
            print("Energy Consumed:")
            print(" min:", np.min(metric_data[1]))
            print(" max:", np.max(metric_data[1]))
            print(" mean:", np.mean(metric_data[1]))
            print("\t ---")
        if prob_type == "custom2":
            print("Energy Consumed:")
            print(" min:", np.min(metric_data[1]))
            print(" max:", np.max(metric_data[1]))
            print(" mean:", np.mean(metric_data[1]))
            print("\t ---")
        if prob_type == "custom3":
            print("Energy Consumed:")
            print(" min:", np.min(metric_data[1]))
            print(" max:", np.max(metric_data[1]))
            print(" mean:", np.mean(metric_data[1]))
            print("\t ---")
        if prob_type == "custom4":
            print("Energy Consumed:")
            print(" min:", np.min(metric_data[1]))
            print(" max:", np.max(metric_data[1]))
            print(" mean:", np.mean(metric_data[1]))
            print("\t ---")
    except:
        pass

    try:
        print("WITH interpol: ")
        print("Success rate= ", interp_metric_data[4])
        print("\t ---")
        print("Travel Time:")
        print(" min:", np.min(interp_metric_data[0]))
        print(" max:", np.max(interp_metric_data[0]))
        print(" mean:", np.mean(interp_metric_data[0]))
        print("\t ---")
        if prob_type == "energy1":
            print("Energy Consumed:")
            print(" min:", np.min(interp_metric_data[1]))
            print(" max:", np.max(interp_metric_data[1]))
            print(" mean:", np.mean(interp_metric_data[1]))
            print("\t ---")
        if prob_type == "energy3":
            print("Net Energy Consumed:")
            print(" min:", np.min(interp_metric_data[2]))
            print(" max:", np.max(interp_metric_data[2]))
            print(" mean:", np.mean(interp_metric_data[2]))
            print("\t ---")
        if prob_type == "energy2":
            print("Net Energy Consumed:")
            print(" min:", np.min(interp_metric_data[3]))
            print(" max:", np.max(interp_metric_data[3]))
            print(" mean:", np.mean(interp_metric_data[3]))
            print("\t ---")
        if prob_type == "custom1":
            print("Net Energy Consumed:")
            print(" min:", np.min(interp_metric_data[2]))
            print(" max:", np.max(interp_metric_data[2]))
            print(" mean:", np.mean(interp_metric_data[2]))
            print("\t ---")
        if prob_type == "custom2":
            print("Net Energy Consumed:")
            print(" min:", np.min(interp_metric_data[2]))
            print(" max:", np.max(interp_metric_data[2]))
            print(" mean:", np.mean(interp_metric_data[2]))
            print("\t ---")
        if prob_type == "custom3":
            print("Net Energy Consumed:")
            print(" min:", np.min(interp_metric_data[2]))
            print(" max:", np.max(interp_metric_data[2]))
            print(" mean:", np.mean(interp_metric_data[2]))
            print("\t ---")
        if prob_type == "custom4":
            print("Net Energy Consumed:")
            print(" min:", np.min(interp_metric_data[2]))
            print(" max:", np.max(interp_metric_data[2]))
            print(" mean:", np.mean(interp_metric_data[2]))
            print("\t ---")
    except:
        pass

 


#  just for plotting traj_data assuming it has already bin saved.
##################################### CHANGED ############# COMMENTED OUT #######################

    print("plotting and saving dynamic plot sequence...")
    plot_interval = 5
    print("with plot interval =", plot_interval)
    interp_traj_data = np.load(join(fpath,'interp_traj_data.npy'),allow_pickle=True)
    interp_metric_data = get_metrics(interp_traj_data)

    # plot_at_ts = [84,128,148,185]
    # plot_at_ts = [28,48,60,82]
    plot_at_ts = [119]

    dynamic_plot_sequence(interp_traj_data, interp_metric_data, g, policy_1d, 
                            vel_field_data, scalar_field_data, nrzns, nrzns_to_plot,
                            fpath, plot_interval, prob_type, plot_at_t=None, fname='Trajectories')

    




# travel_time_list, energy_cons_list, energy_col_list, net_energy_cons_list, success_rate]

    with open(join(fpath,"metrics.txt"), "w") as file1: 
        # Writing data to a file 
        try:
            file1.write("No interpol: ", )
            # file1.write("mean_travel_time, mean_en_cons, mean_net_en_cons, success_rate= ", metrics)
            file1.write("Success rate= " + str(metric_data[4]))
            file1.write("\n ---")
            file1.write("Travel Time:")
            file1.write(" min:"+str( np.min(metric_data[0])))
            file1.write("\n")
            file1.write(" max:"+str( np.max(metric_data[0])))
            file1.write("\n")
            file1.write(" mean:"+str( np.mean(metric_data[0])))
            file1.write("\n ---")
            if prob_type == "energy1":
                file1.write("Energy Consumed:")
                file1.write("\n")
                file1.write(" min:"+str( np.min(metric_data[1])))
                file1.write("\n")
                file1.write(" max:"+str( np.max(metric_data[1])))
                file1.write("\n")
                file1.write(" mean:"+str( np.mean(metric_data[1])))
                file1.write("\n ---")
            if prob_type == "energy2":
                file1.write("Net Energy Consumed:")
                file1.write("\n")
                file1.write(" min:"+str( np.min(metric_data[3])))
                file1.write("\n")
                file1.write(" max:"+str( np.max(metric_data[3])))
                file1.write("\n")
                file1.write(" mean:"+str( np.mean(metric_data[3])))
                file1.write("\n ---")
            if prob_type == "custom1":
                file1.write("Energy Consumed:")
                file1.write("\n")
                file1.write(" min:"+str( np.min(metric_data[1])))
                file1.write("\n")
                file1.write(" max:"+str( np.max(metric_data[1])))
                file1.write("\n")
                file1.write(" mean:"+str( np.mean(metric_data[1])))
                file1.write("\n ---")
            if prob_type == "custom2":
                file1.write("Energy Consumed:")
                file1.write("\n")
                file1.write(" min:"+str( np.min(metric_data[1])))
                file1.write("\n")
                file1.write(" max:"+str( np.max(metric_data[1])))
                file1.write("\n")
                file1.write(" mean:"+str( np.mean(metric_data[1])))
                file1.write("\n ---")
            if prob_type == "custom3":
                file1.write("Energy Consumed:")
                file1.write("\n")
                file1.write(" min:"+str( np.min(metric_data[1])))
                file1.write("\n")
                file1.write(" max:"+str( np.max(metric_data[1])))
                file1.write("\n")
                file1.write(" mean:"+str( np.mean(metric_data[1])))
                file1.write("\n ---")
            if prob_type == "custom4":
                file1.write("Energy Consumed:")
                file1.write("\n")
                file1.write(" min:"+str( np.min(metric_data[1])))
                file1.write("\n")
                file1.write(" max:"+str( np.max(metric_data[1])))
                file1.write("\n")
                file1.write(" mean:"+str( np.mean(metric_data[1])))
                file1.write("\n ---")
        except:
            pass

        try:
            file1.write("WITH interpol: ")
            file1.write("Success rate= "+str( interp_metric_data[4]))
            file1.write("\n ---")
            file1.write("Travel Time:")
            file1.write("\n")
            file1.write(" min:"+str( np.min(interp_metric_data[0])))
            file1.write("\n")
            file1.write(" max:"+str( np.max(interp_metric_data[0])))
            file1.write("\n")
            file1.write(" mean:"+str( np.mean(interp_metric_data[0])))
            file1.write("\n ---")
            if prob_type == "energy1":
                file1.write("Energy Consumed:")
                file1.write(" min:"+str( np.min(interp_metric_data[1])))
                file1.write("\n")
                file1.write(" max:"+str( np.max(interp_metric_data[1])))
                file1.write("\n")
                file1.write(" mean:"+str( np.mean(interp_metric_data[1])))
                file1.write("\n ---")   
            if prob_type == "energy2":
                file1.write("Net Energy Consumed:")
                file1.write(" min:"+str( np.min(interp_metric_data[3])))
                file1.write("\n")
                file1.write(" max:"+str( np.max(interp_metric_data[3])))
                file1.write("\n")
                file1.write(" mean:"+str( np.mean(interp_metric_data[3])))
                file1.write("\n ---")
            if prob_type == "custom1":
                file1.write("Energy Consumed:")
                file1.write(" min:"+str( np.min(interp_metric_data[1])))
                file1.write("\n")
                file1.write(" max:"+str( np.max(interp_metric_data[1])))
                file1.write("\n")
                file1.write(" mean:"+str( np.mean(interp_metric_data[1])))
                file1.write("\n ---")
            if prob_type == "custom2":
                file1.write("Energy Consumed:")
                file1.write(" min:"+str( np.min(interp_metric_data[1])))
                file1.write("\n")
                file1.write(" max:"+str( np.max(interp_metric_data[1])))
                file1.write("\n")
                file1.write(" mean:"+str( np.mean(interp_metric_data[1])))
                file1.write("\n ---")
            if prob_type == "custom2":
                file1.write("Energy Consumed:")
                file1.write(" min:"+str( np.min(interp_metric_data[1])))
                file1.write("\n")
                file1.write(" max:"+str( np.max(interp_metric_data[1])))
                file1.write("\n")
                file1.write(" mean:"+str( np.mean(interp_metric_data[1])))
                file1.write("\n ---")
            if prob_type == "custom4":
                file1.write("Energy Consumed:")
                file1.write(" min:"+str( np.min(interp_metric_data[1])))
                file1.write("\n")
                file1.write(" max:"+str( np.max(interp_metric_data[1])))
                file1.write("\n")
                file1.write(" mean:"+str( np.mean(interp_metric_data[1])))
                file1.write("\n ---")
        except:
            pass

    plot_time = round(time.time() - start_time, 3)
    data_time = round(traj_data_end_time - traj_data_start_time, 3)

    # file contains run time building model and solver time
    file = open(r"temp_runTime.txt","r") 

    # read lines as string from file line by line
    file_lines = file.readlines()
    print(file_lines)
    build_time = float(file_lines[0][0:-1])
    build_time_only = float(file_lines[1][0:-1])
    spvi_time = float(file_lines[2][0:-1])
    
    write_log(interp_metric_data, prob_list, design_param, build_time, spvi_time, plot_time, data_time, build_time_only)

    #find_next_alpha(interp_metric_data, prob_list, design_param, build_time, spvi_time, plot_time)
    




