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

    def f(self , x, t, w):
        term1 = self.eps * sin(w*t) * (x**2)
        term2 = (1 - 2*self.eps*sin(w*t)) * x
        return  term1 + term2 

    def fx(self, x, t, w):
        fx = 2*self.eps*sin(w*t)*(x-1) + 1
        return fx

    def vx_vy(self, x, y, t, w):
        fx = self.fx(x,t,w)
        f = self.f(x,t,w)
        vx = self.A * pi * sin(pi*y) * fx * cos(pi*f) 
        vy = self.A * pi * sin(pi*f) * cos(pi*y)
        return vx, vy

    def sample_w(self, k):
        w_i, w_f = self.w_range
        r = k/(self.n_wsamples - 1)
        w = w_i + r*(w_f - w_i)
        return w

    def get_ij_from_xy(self, i, j):
        x = self.xs[j]
        y = self.ys[self.gsize - 1 - i]
        return x, y

    def generate_R(self):
        # Reealisation matrix
        R_vx = np.empty((self.nt, self.n,self.n_wsamples))
        R_vy = np.empty((self.nt, self.n,self.n_wsamples))
        assert(R_vx.shape == (self.nt,self.n, self.n_wsamples))
        
        for t in range(self.nt):
            for k in range(self.n_wsamples):
                for i in range(self.gsize):
                    for j in range(self.gsize):
                        row = self.gsize*i + j
                        col = k
                        x, y = self.get_ij_from_xy(i,j)
                        w = self.sample_w(k)
                        vx, vy = self.vx_vy(x, y, t, w)
                        R_vx[t, row, col] = vx
                        R_vy[t, row, col] = vy
        return R_vx, R_vy

    def compute_modes_and_coeffs(self):
        Rx, Ry = self.generate_R()
        u, S, vh = np.linalg.svd(Rx)
        return 


gsize = 50
nt = 10
dxy = 2/gsize
A = 0.1
eps = 0.1
op_nrzns = 10
n_wsamples = 10
w_range = ( pi/10, 8*pi/10 )

dg1 = DG_velocity_field(gsize, nt, dxy, A, eps, op_nrzns, n_wsamples, w_range)
Rx, Ry = dg1.generate_R()
print(Rx.shape)

X,Y = np.meshgrid(dg1.xs, dg1.ys)
print(X, Y)

t = 0
r = 0
vx = np.reshape(Rx[t,:,r],(gsize,gsize))
vy = np.reshape(Ry[t,:,r],(gsize,gsize))
print(vx.shape)
# plt.quiver(X, Y, vx, vy, color = 'c', alpha = 0.5)
strm = plt.streamplot(X, Y, vx, vy, color=vx, linewidth=2, cmap='autumn')
plt.savefig('test')






# # all jet velocity field rotates direction from north to east at half time
# # veritical strip of scalar field moves rightwards with time

# all_u_mat = np.zeros((nt, gsize, gsize), dtype = np.float32)
# all_v_mat = np.zeros((nt, gsize, gsize), dtype = np.float32)
# all_ui_mat = np.ones((nt, nmodes, gsize, gsize), dtype = np.float32)
# all_vi_mat = np.zeros((nt, nmodes, gsize, gsize), dtype = np.float32)
# all_Yi = np.zeros((nt, nrzns, nmodes), dtype = np.float32)

# all_s_mat = np.zeros((nt, gsize, gsize), dtype = np.float32)
# obstacle_mask = np.zeros((nt, gsize, gsize), dtype = np.int32)


# mid = int(0.4*nrzns)
# all_Yi[:, mid:nrzns, :] = 1
# all_Yi[:, 0:mid, :] = 1.2

# # for t in range(nt):
# #     for r in range(nrzns):
# #         if r==0:
# #             print("r = ", r)
# #             for i in range(gsize):
# #                 for j in range(gsize):
# #                     # print(extract_velocity(vel_field_data, t, i ,j ,r)[0], end =" ")
# #                     print(all_s_mat[t,i,j], end=" ")
# #                 print()
# #             print("\n\n")

# vel_field_data = [ all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, all_Yi ]
# scalar_field_data = [all_s_mat, obstacle_mask]
# files = ["all_u_mat.npy", "all_v_mat.npy", "all_ui_mat.npy", "all_vi_mat.npy", "all_Yi.npy"]
# scalar_files = ["all_s_mat.npy", "obstacle_mask.npy"]

# for i in range(5):
#     np.save(files[i], vel_field_data[i])
#     print("Saved ", files[i])
# for i in range(len(scalar_field_data)):
#     np.save(scalar_files[i], scalar_field_data[i])
#     print("Saved ", scalar_files[i])

# print("\n ----- Saved field files! ------\n")

