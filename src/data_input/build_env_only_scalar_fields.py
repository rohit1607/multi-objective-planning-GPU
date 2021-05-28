import numpy as np 
from math import cos, sin, pi, floor
import matplotlib.pyplot as plt
import imageio
from scipy.interpolate import griddata







class DG_scalar_field:

    def __init__(self, gsize, nt, dt, dxy, A, eps, op_nrzns, n_wsamples, w_range, interpolate_degree=1):
        self.gsize = gsize
        self.n = gsize*gsize
        self.nt = nt
        self.dt = dt
        self.op_nrzns = op_nrzns
        self.n_wsamples = n_wsamples
        self.eps = eps
        self.A = A
        self.w_range = w_range
        self.dxy = dxy
        self.xs = self.get_xs_ys(dxy)
        self.ys = self.get_xs_ys(dxy)
        self.ts = [i*dt for i in range(nt)]
        self.interpolate_degree = interpolate_degree
        print("self.dxy= ", self.dxy)
        print("self.xs[-1]= ", self.xs[-1])
        print("self.dt = ", dt, nt, self.ts[-1])

    def get_xs_ys(self,dxy):
        return [(0.5 + i)*dxy for i in range(self.gsize)]

    def f(self , x, t):
        w = pi/2
        # length = len(self.xs)
        # term1 = self.eps * sin(w*t) * (x**2)
        # term2 = (1 - 2*self.eps*sin(w*t)) * x
        T = t*self.interpolate_degree/2
        l = self.xs[-1]/2
        high_val = 1
        low_val = 0
        dom_size = self.xs[-1] + self.dxy
        if ((x + (T*self.dxy)) <= l):
            return high_val
        elif (l < (x + (T*self.dxy)) < l+(0.2*dom_size)):
            # return -0.5*(x + (T*self.dxy)) + 5
            slope = (high_val-low_val)/(-0.2*dom_size)
            return high_val + (slope*(x - (l- T*self.dxy)))
        else:
            return low_val


    def phi(self, x, y, t):
        # *sin(2*pi*t/20))
        # return self.A*(1+ sin(0.25*pi*self.f(x,t))*cos(0.05*pi*(y+1)))
        return self.A*(self.f(x,t))
        
    def sample_w(self, k):
        # not used when generating only mean scalar field
        w_i, w_f = self.w_range
        r = k/(self.n_wsamples - 1)
        w = w_i + r*(w_f - w_i)
        # if k==1:
        #     print("(k,w)=",(k,w*10/pi))
        return w

    def get_ij_from_xy(self, i, j):
        x = self.xs[j]
        y = self.ys[self.gsize - 1 - i]
        return x, y

    def generate_phi(self):
        # generate mean scalar field
        phi = np.empty((self.nt, self.n),  dtype = np.float32)
        for tid in range(self.nt):
            # t = self.ts[tid]
            t= tid
            for i in range(self.gsize):
                for j in range(self.gsize):
                    col = self.gsize*i + j
                    x, y = self.get_ij_from_xy(i,j)
                    phi[tid, col] = self.phi(x,y,t)

        g= self.gsize
        phi_grid = np.empty((self.nt,g,g), dtype = np.float32)
        for tid in range(self.nt):
            phi_grid[tid,:,:]= np.resize(phi[tid,:],(g,g) )

        return phi_grid




def fill_obstacles(obstacle_mask, final_gsize, nt, dyn_obstacle_data):
    # row/col up/down init/final
    # obs_upper_row, obs_left_col, obs_width have range 0 to final_gsize
    # obs_speed = [vx, vy] units= cells/timestep

    mode = dyn_obstacle_data[-1]

    if mode == 'None':
        obstacle_mask[:, :, :] = 0

    if mode == 'static':
        ru = int(0.4*final_gsize)
        rd = int(0.5*final_gsize)
        cl = int(0.5*final_gsize)
        cr = int(0.6*final_gsize)
        obstacle_mask[:, ru:rd, cl:cr] = 1

    elif mode == 'dynamic':
        obs_upper_row, obs_left_col, obs_width, obs_speed, _ = dyn_obstacle_data
        assert(0<obs_left_col<final_gsize and 0<obs_upper_row<final_gsize)
        ru_i = obs_upper_row
        # rd_i = ru_i + obs_width
        cl_i = obs_left_col
        # cr_i = cl_i + obs_width
        for t in range(nt):
            cl = floor(cl_i + (t * obs_speed[0]))
            cr = int(cl + obs_width)
            ru = floor(ru_i + (t * obs_speed[1]))
            rd = int(ru + obs_width)
            # if (cl == 0 or cl == final_gsize -1 or cr == 0 or cr == final_gsize -1 or ru == 0 or ru == final_gsize -1  or rd == 0 or rd == final_gsize -1):
            # if obstacle is in domain

            if (0 < cl < final_gsize-1) and (0 < cr < final_gsize-1) and (0 < ru < final_gsize-1) and (0 < rd < final_gsize-1):
                obstacle_mask[t, ru:rd, cl:cr] = 1
            else:
                obstacle_mask[t, :, :] = 0

    elif mode == 'multiple_dynamic':
        obs_upper_row, obs_left_col, obs_width, obs_speed, _ = dyn_obstacle_data
        assert(0<obs_left_col<final_gsize and 0<obs_upper_row<final_gsize)
        ru_i = [obs_upper_row, obs_upper_row] #both obstacles along the same row
        # rd_i = ru_i + obs_width
        cl_i = [obs_left_col, obs_left_col + (final_gsize/2)]
        # cr_i = cl_i + obs_width
        num_obs = len(ru_i)
        for t in range(nt):
            for k in range(num_obs):
                cl = floor(cl_i[k] + (t * obs_speed[0]))%final_gsize
                cr = int(cl + obs_width)%final_gsize
                ru = floor(ru_i[k] + (t * obs_speed[1]))%final_gsize
                rd = int(ru + obs_width)%final_gsize
                # if (cl == 0 or cl == final_gsize -1 or cr == 0 or cr == final_gsize -1 or ru == 0 or ru == final_gsize -1  or rd == 0 or rd == final_gsize -1):
                # if obstacle is in domain
                
                # if (0 < cl < final_gsize-1) and (0 < cr < final_gsize-1) and (0 < ru < final_gsize-1) and (0 < rd < final_gsize-1):
                #     obstacle_mask[t, ru:rd, cl:cr] = 1
                # else:
                #     obstacle_mask[t, :, :] = 0

                if (cl < cr) and (ru < rd):
                    obstacle_mask[t, ru:rd, cl:cr] = 1
                elif (cl > cr) and (ru < rd):
                    obstacle_mask[t, ru:rd, cl:] = 1
                    obstacle_mask[t, ru:rd, 0:cr] = 1
                elif (cl < cr) and (ru > rd):
                    obstacle_mask[t, ru:, cl:cr] = 1
                    obstacle_mask[t, 0:rd, cl:cr] = 1
                elif (cl > cr) and (ru > rd):
                    obstacle_mask[t, ru:, cl:] = 1
                    obstacle_mask[t, 0:rd, cl:] = 1
                    obstacle_mask[t, ru:, 0:cr] = 1
                    obstacle_mask[t, 0:rd, 0:cr] = 1



    return obstacle_mask 


def test_scalar_field(all_s_mat, X, Y, nt):
    fig = plt.figure(figsize=(10, 10))
    images = []
    fname = "test.png"

    for t in range(nt):
        s_arr = all_s_mat[t,:,:]
        plt.contourf(X, Y, s_arr, cmap = "YlOrRd", alpha = 0.5, zorder = -1e5)
        plt.colorbar()
        plt.savefig(fname)
        plt.clf()
        images.append(imageio.imread(fname))
    imageio.mimsave('scalar_field.gif', images, duration = 0.5)


# init_gsize = 25
# nt = 10
# dt = 10/nt
# dxy = 2/init_gsize
# A = 0.5
# A_sc = 3
# eps = 0.1
# op_nrzns = 10
# n_wsamples = 1000
# w_range = ( pi/10, 8*pi/10 )
# wy = 0.5*pi
# wx = pi
# # interpolates 
# interpolate_degree = 8
# n_modes = 3

def build_scalar_fields(init_gsize, interpolate_degree, nt, dt, dxy, A_sc, eps,
                                                op_nrzns, n_wsamples, w_range, wx,  
                                                dyn_obstacle_data = None):
    wy = 0.5*pi
    final_gsize = init_gsize*interpolate_degree
    print("final gsize=", final_gsize)
    print("nt=", nt)
    print("nr=", n_wsamples)
    print("dxy=",dxy)

    print("initialising objects")
    cloud = DG_scalar_field(final_gsize, nt, dt, dxy/interpolate_degree, A_sc, eps, op_nrzns, n_wsamples, w_range)

    print("generating scalar field")
    all_s_mat = cloud.generate_phi()
    assert(all_s_mat.shape == (nt,final_gsize,final_gsize))
    # X,Y = np.meshgrid(np.array(cloud.xs)/interpolate_degree, np.flip(np.array(cloud.ys)/interpolate_degree))
    X,Y = np.meshgrid(np.array(cloud.xs), np.flip(np.array(cloud.ys)))
    test_scalar_field(all_s_mat, X, Y, nt)
    # print(C.shape, M.shape, R_mean.shape)

    obstacle_mask = np.zeros((nt, final_gsize, final_gsize), dtype = np.int32)
    obstacle_mask = fill_obstacles(obstacle_mask, final_gsize, nt, dyn_obstacle_data)

    scalar_field_data = [all_s_mat, obstacle_mask]
    scalar_files = ["all_s_mat.npy", "obstacle_mask.npy"]


    for i in range(len(scalar_field_data)):
        np.save(scalar_files[i], scalar_field_data[i])
        print("Saved ", scalar_files[i],"shape= ", scalar_field_data[i].shape,
                                    " type=", scalar_field_data[i].dtype,
                                    "max_val=", np.max(scalar_field_data[i]))

    print("\n ----- Saved field files! ------\n")

