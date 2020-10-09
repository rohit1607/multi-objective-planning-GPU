import numpy as np 
from math import cos, sin, pi
import matplotlib.pyplot as plt
import imageio

def extract_velocity(vel_field_data, t, i, j, rzn):
    # all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, all_Yi = vel_field_data
    #    0          1           2           3           4
    # vx = all_u_mat[t,i,j] + np.matmul(all_ui_mat[t, :, i,j], all_Yi[t, rzn,:])
    # vy = all_v_mat[t,i,j] + np.matmul(all_vi_mat[t, :, i,j], all_Yi[t, rzn,:])
    vx = vel_field_data[0][t, i, j] + np.matmul(vel_field_data[2][t, :, i, j],vel_field_data[4][t, rzn,:])
    vy = vel_field_data[1][t, i, j] + np.matmul(vel_field_data[3][t, :, i, j], vel_field_data[4][t, rzn,:])

    return vx,vy

class DG_velocity_field:

    def __init__(self, gsize, nt, dxy, A, eps, op_nrzns, n_wsamples, w_range):
        self.gsize = gsize
        self.n = gsize*gsize
        self.nt = nt
        self.op_nrzns = op_nrzns
        self.n_wsamples = n_wsamples
        self.eps = eps
        self.A = A
        self.w_range = w_range
        self.dxy = dxy
        self.xs = [(0.5 + i)*dxy for i in range(gsize)]
        self.ys = [(0.5 + i)*dxy for i in range(gsize)]



    def phi(self, x, y, t, w):
        return ((x-1)**2) + ((y-1)**2) + t

    def vx_vy(self, x, y, t, w):
        vx = -2*(y-1)*(w/w+1)
        vy = 2*(x-1)*(w/w+1)
        return vx, vy

    def sample_w(self, k):
        w_i, w_f = self.w_range
        r = k/(self.n_wsamples - 1)
        w = w_i + r*(w_f - w_i)
        if k==1:
            print("(k,w)=",(k,w*10/pi))
        return w

    def get_ij_from_xy(self, i, j):
        x = self.xs[j]
        y = self.ys[self.gsize - 1 - i]
        return x, y

    def generate_R(self):
        # Reealisation matrix
        R_vx = np.empty((self.nt, self.n_wsamples, self.n))
        R_vy = np.empty((self.nt, self.n_wsamples, self.n))
        phi = np.empty((self.nt, self.n_wsamples, self.n))
        assert(R_vx.shape == (self.nt, self.n_wsamples, self.n))
        
        for t in range(self.nt):
            for k in range(self.n_wsamples):
                w = self.sample_w(k)
                for i in range(self.gsize):
                    for j in range(self.gsize):
                        row = k
                        col = self.gsize*i + j
                        x, y = self.get_ij_from_xy(i,j)
                        vx, vy = self.vx_vy(x, y, t, w)
                        R_vx[t, row, col] = vx
                        R_vy[t, row, col] = vy
                        phi[t, row, col] = self.phi(x,y,t,w)
        return R_vx, R_vy, phi

    def rank_reduction(self, u, s, vh, n_modes):
        u1 = u[:, 0:n_modes]
        s1_mat = np.diag(s[0:n_modes])
        vh1 = vh[0:n_modes, :]
        C = np.matmul(u1, s1_mat)
        M = vh1
        assert(C.shape == (self.n_wsamples, n_modes), print(C.shape,(self.n_wsamples, n_modes)))
        assert(M.shape == (n_modes, self.n*2))
        return C,M

    def compute_modes_and_coeffs(self, n_modes):
        Rx, Ry, phi = self.generate_R()
        R = np.concatenate((Rx, Ry), axis = 2)

        R_mean = np.mean(R, axis =1)
        R_mean_full = np.empty_like(R)
        for k in range(self.n_wsamples):
            R_mean_full[:,k,:] = R_mean    

        assert(R.shape == (self.nt, self.n_wsamples, self.n*2))
        assert(R_mean_full.shape == R.shape)
        assert(R_mean.shape == (self.nt, self.n*2))

        X = R - R_mean_full
        print("n_modes= ", n_modes)
        # n_modes = n_modes -19
        C = np.empty((self.nt, self.n_wsamples, n_modes))
        M = np.empty((self.nt, n_modes, self.n*2))
        sn = 5
        print("first ",sn, "singular valus ")
        for t in range(self.nt):
            u, s, vh = np.linalg.svd(X[t,:,:])
            print(" out of ",len(s),"at t",t, ": \n", np.round(s[0:sn],2),"\n")
            Ct, Mt = self.rank_reduction(u,s,vh, n_modes)
            C[t,:,:] = Ct
            M[t,:,:] = Mt 
        return C, M, R_mean_full






def test_generate_R_plots(dg1):
    Rx, Ry, phi = dg1.generate_R()
    print(Rx.shape)

    X,Y = np.meshgrid(dg1.xs, np.flip(dg1.ys))
    print(X,"\n", Y)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0,2)
    ax.set_ylim(0,2)
    plt.gca().set_aspect('equal', adjustable='box')
    for r in range(dg1.n_wsamples):
        images= []
        for t in range(nt):
            vx = np.reshape(Rx[t,r,:],(gsize,gsize))
            vy = np.reshape(Ry[t,r,:],(gsize,gsize))
            phi_tr = np.reshape(phi[t,r,:],(gsize,gsize))
            # plt.quiver(X, Y, vx, vy, color = 'c', alpha = 0.5)
            plt.streamplot(X, Y, vx, vy, color='k', linewidth=2)
            plt.contourf(X, Y, phi_tr, cmap='RdBu')
            plt.colorbar()
            fname = "test.png"
            plt.savefig(fname)
            plt.clf()
            images.append(imageio.imread(fname))
        gifname = "movie" + "_r" + str(r) + ".gif"
        imageio.mimsave(gifname, images, duration = 1)


def test_rank_reduced_field(dg1):
    # n_modes = np.min((self.n*2, self.n_wsamples)) #criteria
    n_modes = 8
    C, M, R_mean_full = dg1.compute_modes_and_coeffs(n_modes)
    t = 10
    Ct, Mt = C[t,:,:], M[t,:,:]
    mean_Rt = R_mean_full[t,:,:]
    red_Xt = np.matmul(Ct, Mt)
    red_Rt = mean_Rt + red_Xt
    assert(red_Xt.shape == mean_Rt.shape)
    print("red_Rt.shape = ",red_Rt.shape)

    old_Rx, old_Ry,phi = dg1.generate_R()
    old_Rx_t = old_Rx[t,:,:]
    old_Ry_t = old_Ry[t,:,:]

    old_Rt = np.concatenate((old_Rx_t, old_Ry_t), axis = 1)
    assert(old_Rt.shape == (dg1.n_wsamples, dg1.n*2))
    assert(old_Rt.shape == red_Rt.shape)
    print("arrays are nearly equal: ", np.allclose(red_Rt, old_Rt))

    k = 1  #kth realisation
    red_Rx_k = red_Rt[k,0:dg1.n]
    red_Ry_k = red_Rt[k,dg1.n:dg1.n*2]
    print(red_Rx_k.shape, red_Ry_k.shape)
    red_vx =np.reshape(red_Rx_k, (dg1.gsize, dg1.gsize))
    red_vy =np.reshape(red_Ry_k, (dg1.gsize, dg1.gsize))
    X,Y = np.meshgrid(dg1.xs, dg1.ys)
    phi_tr = np.reshape(phi[t,k,:], (dg1.gsize, dg1.gsize))
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0,2)
    ax.set_ylim(0,2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.streamplot(X, Y, red_vx, red_vy, color='k', linewidth=2)
    plt.contourf(X, Y, phi_tr, cmap='RdBu')
    plt.colorbar()
    plt.savefig("reduced_rank"+"t"+str(t))
    return

def plot_modes_of_rand_reduced_field(dg):
    n_modes = 5
    assert(n_modes <= dg.n_wsamples)
    C, M, R_mean_full = dg.compute_modes_and_coeffs(n_modes)
    fig = plt.figure(figsize=(10, 10))
    t = 9
    Ct, Mt = C[t,:,:], M[t,:,:]
    X,Y = np.meshgrid(dg.xs, dg.ys)

    for k in range(4):
        ax = fig.add_subplot(2, 2, k+1)
        Mt_k = Mt[k,:]
        mode_vx = np.reshape(Mt_k[0:dg.n], (dg.gsize,dg.gsize))
        mode_vy = np.reshape(Mt_k[dg.n:dg.n*2], (dg.gsize,dg.gsize))
        ax.streamplot(X, Y, mode_vx, mode_vy, color='k', linewidth=2)
    
    plt.savefig('modes')



    





gsize = 50
nt = 11
dxy = 2/gsize
A = 0.1
eps = 0.1
op_nrzns = 10
n_wsamples = 8
w_range = ( pi/10, 8*pi/10 )

dg1 = DG_velocity_field(gsize, nt, dxy, A, eps, op_nrzns, n_wsamples, w_range)
# C, M = dg1.compute_modes_and_coeffs()
# test_rank_reduced_field(dg1)
test_generate_R_plots(dg1)
# plot_modes_of_rand_reduced_field(dg1)
