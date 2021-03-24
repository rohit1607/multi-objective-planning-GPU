import numpy as np 
from math import cos, sin, pi
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imageio
from scipy.interpolate import griddata
from os.path import join
import os,sys,inspect
from pathlib import Path
import pandas as pd
import seaborn as sns



currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 



# print(prob_path)
class plot_environment_fields:
    def __init__(self, prob_path):
        self.all_u_mat = np.load(join(prob_path, "all_u_mat.npy"))
        self.all_ui_mat = np.load(join(prob_path, "all_ui_mat.npy"))
        self.all_v_mat = np.load(join(prob_path, "all_v_mat.npy"))
        self.all_vi_mat = np.load(join(prob_path, "all_vi_mat.npy"))
        self.all_Yi = np.load(join(prob_path, "all_Yi.npy"))
        self.all_s_mat = np.load(join(prob_path, "all_s_mat.npy"))
        self.obstacle_mask = np.load(join(prob_path, "obstacle_mask.npy"))
        try:
            self.all_phi_mat = np.load(join(prob_path, "all_phi_mat.npy"))
            self.all_phi_i_mat = np.load(join(prob_path, "all_phi_i_mat.npy"))
            self.all_phi_Yi_mat = np.load(join(prob_path, "all_phi_Yi_mat.npy"))
            print(self.all_phi_i_mat.shape)
        except:
            pass
        print("all_Yi.shape =",self.all_Yi.shape)
        print("all_s_mat.shape =",self.all_s_mat.shape)

        self.nt, self.nmodes, self.gsize, _ = self.all_ui_mat.shape
        self.dxy = 2/self.gsize
        self.xs = np.array([(0.5+i)*self.dxy for i in range(self.gsize)])
        self.ys = self.xs
        self.X, self.Y = np.meshgrid(self.xs, np.flip(self.ys))
        self.prob_path = prob_path
        self.num_fontsize = 20
        self.strmplot_arrowsize  = 3.0 # scaling factor
        self.strmplot_lw = 2 # linewidth of streamplot

    def plot_modes(self, t, show_contours="mode_vel_mag", show_colorbar=False, show_fig=True, save_fig=False):
        # t is the time index at which the plot is made
        fig = plt.figure(figsize=(10, 10))
        gs1 = gridspec.GridSpec(2, 2)
        gs1.update(wspace=0.05, hspace=0.15) # set the spacing between axes. 
 
        for k in range(4): #hardcoded for first 4 modes
            # ax = fig.add_subplot(2, 2, k+1)
            title = "Mode " + str(k+1)
            ax = plt.subplot(gs1[k])
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            ax.axes.set_title(title, fontsize=22)
            plt.streamplot(self.X, self.Y, self.all_ui_mat[t,k,:,:], self.all_vi_mat[t,k,:,:],color='k', 
                            linewidth=self.strmplot_lw, arrowsize=0.5*self.strmplot_arrowsize)
            if show_contours == "phi":
                plt.contourf(self.X, self.Y, self.all_phi_i_mat[t,k,:,:], cmap='bwr')
            if show_contours == "mode_vel_mag":
                u_t_k = self.all_ui_mat[t,k,:,:]
                v_t_k = self.all_vi_mat[t,k,:,:]
                vel_mag_t_k = (u_t_k**2 + v_t_k**2)**0.5 #velocity magnitude at t,k
                plt.contourf(self.X, self.Y, vel_mag_t_k, cmap='Blues')
            if show_colorbar==True:
                cbar = plt.colorbar()
                cbar.ax.tick_params(labelsize=self.num_fontsize) 

        if save_fig:
            plt.savefig(join(self.prob_path, "modes@") + str(t),bbox_inches = "tight", dp = 300)
        if show_fig:
            plt.show()

    def plot_mean_vel_field(self, t, k,  show_obstacle=False, 
                                    show_streams = True, 
                                    show_contours = None,
                                    show_colorbar = False,
                                    show_fig = True,
                                    save_fig = False):
        # t is the time index at which the plot is made
        # k is the rzn index

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        cb=''
        self.setup_grid_in_plot(fig, ax)

        u_t_k = self.all_u_mat[t,:,:] 
        v_t_k = self.all_v_mat[t,:,:] 
        try:
            phi_t_k = self.all_phi_mat[t,:,:]
        except:
            pass

        if show_streams:
            plt.streamplot(self.X, self.Y, u_t_k, v_t_k,color='k', linewidth=self.strmplot_lw, arrowsize=self.strmplot_arrowsize)
        if show_contours != None:
            if show_contours == "phi":
                plt.contourf(self.X, self.Y, phi_t_k, cmap='bwr')
            if show_contours == "vel_mag":
                vel_mag_t_k = (u_t_k**2 + v_t_k**2)**0.5 #velocity magnitude at t,k
                plt.contourf(self.X, self.Y, vel_mag_t_k, cmap = 'Blues')
        if show_colorbar:
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=self.num_fontsize) 
            cb = 'wcb'
        if show_obstacle:
            self.plot_obstacle(t)

        if save_fig:
            plt.savefig(join(self.prob_path, "mean_vel_field_") + cb + "@" + str(t) + "_" + str(k),bbox_inches = "tight", dp = 300)
        if show_fig:
            plt.show()



    def plot_vel_field(self, t, k,  show_obstacle=False, 
                                    show_streams = True, 
                                    show_contours = None,
                                    show_colorbar = False,
                                    show_fig = True,
                                    save_fig = False):
        # t is the time index at which the plot is made
        # k is the rzn index

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        matmul_ui_Yi_t = 0
        matmul_vi_Yi_t = 0
        matmul_phi_i_phi_Yi_t = 0
        cb=''
        self.setup_grid_in_plot(fig, ax)

        for m in range(self.nmodes):
            matmul_ui_Yi_t += (self.all_ui_mat[t,m,:,:]*self.all_Yi[t,k,m])
            matmul_vi_Yi_t += (self.all_vi_mat[t,m,:,:]*self.all_Yi[t,k,m])
            u_t_k = self.all_u_mat[t,:,:] + matmul_ui_Yi_t
            v_t_k = self.all_v_mat[t,:,:] + matmul_vi_Yi_t
            try:
                matmul_phi_i_phi_Yi_t += (self.all_phi_i_mat[t,m,:,:]*self.all_phi_Yi_mat[t,k,m])
                phi_t_k = self.all_phi_mat[t,:,:] + matmul_phi_i_phi_Yi_t
            except:
                pass

      

        if show_streams:
            plt.streamplot(self.X, self.Y, u_t_k, v_t_k, color='k', linewidth=self.strmplot_lw, arrowsize=self.strmplot_arrowsize)
        if show_contours != None:
            if show_contours == "phi":
                plt.contourf(self.X, self.Y, phi_t_k, cmap='bwr')
            if show_contours == "vel_mag":
                vel_mag_t_k = (u_t_k**2 + v_t_k**2)**0.5 #velocity magnitude at t,k
                plt.contourf(self.X, self.Y, vel_mag_t_k, cmap = 'Blues')
        if show_colorbar:
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=self.num_fontsize) 
            cb = 'wcb'
        if show_obstacle:
            self.plot_obstacle(t)

        if save_fig:
            plt.savefig(join(self.prob_path, "vel_field_") + cb + "@" + str(t) + "_" + str(k),bbox_inches = "tight", dp = 300)
        if show_fig:
            plt.show()


    def plot_scalar_field(self, t, k, show_obstacle=False,
                                        show_colorbar=False,
                                        show_fig=True,
                                        save_fig=False):

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        self.setup_grid_in_plot(fig, ax)
        cb =''
        plt.contourf(self.X, self.Y, self.all_s_mat[t,:,:], cmap = "YlOrRd_r", alpha = 0.5)
        if show_obstacle:
            self.plot_obstacle(t)
        if show_colorbar:
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=self.num_fontsize) 
            cb = 'wcb'
        if save_fig:
            plt.savefig(join(self.prob_path, "scalar_field_") + cb + "@" + str(t) + "_" + str(k),bbox_inches = "tight",dp = 300)
        if show_fig:
            plt.show()

    def get_cell_corners(self,t,i,j):
        xc = self.xs[j]
        yc = self.ys[self.gsize - 1 - i]
        xl = xc - self.dxy/2
        xr = xc + self.dxy/2
        yt = yc + self.dxy/2
        yb = yc - self.dxy/2
        corner_x_coords = [xl, xr, xr, xl]
        corner_y_coords = [yt, yt, yb, yb]
        return corner_x_coords, corner_y_coords

    def plot_obstacle(self, t):
        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(1, 1, 1)
        for i in range(self.gsize):
            for j in range(self.gsize):
                if self.obstacle_mask[t,i,j] == 1:
                    x_corners, y_corners = self.get_cell_corners(t,i,j)
                    plt.fill(x_corners, y_corners, 'dimgrey', alpha = 1, zorder = 1e6)
        # plt.show()
        return

    def plot_all(self,t,k, show_fig=True, save_fig=False):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        self.setup_grid_in_plot(fig, ax)

        plt.contourf(self.X, self.Y, self.all_s_mat[t,:,:], cmap = "YlOrRd_r", alpha = 0.5)
        matmul_ui_Yi_t = 0
        matmul_vi_Yi_t = 0
        matmul_phi_i_phi_Yi_t = 0
     

        for m in range(self.nmodes):
            matmul_ui_Yi_t += (self.all_ui_mat[t,m,:,:]*self.all_Yi[t,k,m])
            matmul_vi_Yi_t += (self.all_vi_mat[t,m,:,:]*self.all_Yi[t,k,m])
        u_t_k = self.all_u_mat[t,:,:] + matmul_ui_Yi_t
        v_t_k = self.all_v_mat[t,:,:] + matmul_vi_Yi_t
        
        try:
            matmul_phi_i_phi_Yi_t += (self.all_phi_i_mat[t,m,:,:]*self.all_phi_Yi_mat[t,k,m])
            phi_t_k = self.all_phi_mat[t,:,:] + matmul_phi_i_phi_Yi_t
        except:
            pass
     
        plt.streamplot(self.X, self.Y, u_t_k, v_t_k,color='k', linewidth=self.strmplot_lw, arrowsize=self.strmplot_arrowsize)
        self.plot_obstacle(t)
        if save_fig:
            plt.savefig(join(self.prob_path, "environment@") + str(t) + "_" + str(k),bbox_inches = "tight", dp =300)
        if show_fig:
            plt.show()


    def plot_env_gif(self):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        # self.setup_grid_in_plot(fig, ax)
        images = []
        fname = "last_fig_in_gif.png"

        for t in range(self.nt):
            plt.contourf(self.X, self.Y, self.all_s_mat[t,:,:], cmap = "YlOrRd_r", alpha = 0.5)
            self.plot_obstacle(t)
            plt.savefig(fname)
            plt.clf()
            images.append(imageio.imread(fname))
        imageio.mimsave(join(self.prob_path, 'scalar_field_dynObs.gif'), images, duration = 0.5)


    def plot_env_sequence(self, k, plot_interval, show_contours = "vel_mag"):

        plot_seq_path = join(self.prob_path, "plot_sequence_env")
        Path(plot_seq_path).mkdir(parents=False, exist_ok=True)
        print("plotting env sequence at tid:")
        for t in range(self.nt):
            if (t%plot_interval == 0 or t == self.nt -1):
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(1, 1, 1)
                self.setup_grid_in_plot(fig, ax)
                # title = "bg_"+ show_contours+ " t = " + str(t)
                # plt.title(title)

                print(t, end='\t')
                matmul_ui_Yi_t = 0
                matmul_vi_Yi_t = 0
                matmul_phi_i_phi_Yi_t = 0
            
                for m in range(self.nmodes):
                    matmul_ui_Yi_t += (self.all_ui_mat[t,m,:,:]*self.all_Yi[t,k,m])
                    matmul_vi_Yi_t += (self.all_vi_mat[t,m,:,:]*self.all_Yi[t,k,m])
                   
                u_t_k = self.all_u_mat[t,:,:] + matmul_ui_Yi_t
                v_t_k = self.all_v_mat[t,:,:] + matmul_vi_Yi_t

                try:
                    matmul_phi_i_phi_Yi_t += (self.all_phi_i_mat[t,m,:,:]*self.all_phi_Yi_mat[t,k,m])
                # phi_t_k = self.all_phi_mat[t,:,:] + matmul_phi_i_phi_Yi_t
                except:
                    pass
            
                plt.streamplot(self.X, self.Y, u_t_k, v_t_k,color='k', linewidth=self.strmplot_lw, arrowsize=self.strmplot_arrowsize, arrowstyle='->')
                if show_contours != None:
                    if show_contours == "scalar_field":
                        plt.contourf(self.X, self.Y, self.all_s_mat[t,:,:], cmap = "YlOrRd_r", alpha = 0.5)
                    if show_contours == "vel_mag":
                        vel_mag_t_k = (u_t_k**2 + v_t_k**2)**0.5 #velocity magnitude at t,k
                        plt.contourf(self.X, self.Y, vel_mag_t_k, cmap = 'Blues')
                # cbar = plt.colorbar()
                # cbar.ax.tick_params(labelsize=self.num_fontsize) 

                self.plot_obstacle(t)
                fname_pfx = "bg_"+ show_contours+ "env"
                fname = join(plot_seq_path, fname_pfx) + "@t" + str(t) + ".png"
                plt.savefig(fname,bbox_inches = "tight", dpi =300)
                plt.clf()
                plt.close()


    def setup_grid_in_plot(self, fig, ax):
        ax.set_xlim(0,self.xs[-1] + (self.dxy/2))
        ax.set_ylim(0,self.ys[-1] + (self.dxy/2))

        minor_ticks = [i*self.dxy/1 for i in range(0, self.gsize + 1, 10)]
        major_ticks = [i*self.dxy/1 for i in range(0, self.gsize + 1, 40)]

        ax.set_xticks(minor_ticks, minor=True)
        ax.set_xticks(major_ticks, minor=False)
        ax.set_yticks(major_ticks, minor=False)
        ax.set_yticks(minor_ticks, minor=True)

        # ax.grid(b= True, which='both', color='#CCCCCC', axis='both',linestyle = '-', alpha = 0.5, zorder = -1e6)
        ax.tick_params(axis='both', which='both', labelsize=self.num_fontsize)

        ax.set_xlabel('X (Non-Dim)', fontsize=22)
        ax.set_ylabel('Y (Non-Dim)', fontsize=22)

    def plot_coefs(self, rzn_list, save_fig= True, show_fig=True):
        # rzn_list is a list of rzn_ids for which you want to plot the coeffs
        fig = plt.figure(figsize=(40, 10))
        
        colors = ['b','g','r','y','c']
        # linestyles = ['-', '--', ':', '-.']
        labels = ['1st', '2nd', '3rd', '4th', '5th']
        labels = [label + ' mode' for label in labels]
        for i in range(len(rzn_list)):
            ax = fig.add_subplot(2, 4, i+1)
            rzn_id = rzn_list[i]
            for m in range(4):
                ax.plot(self.all_Yi[:,rzn_id, m], color = colors[m],label = labels[m], linewidth=4)
                ax.legend(fontsize = 16)
                # ax.set_title('rzn id = ' + str(rzn_id))
                ax.tick_params(axis='both', which='both', labelsize=self.num_fontsize)
                ax.set_xlabel('t', fontsize=22)
                ax.set_ylabel('Y', fontsize=22)
                # ax.xaxis.set_label_coords(1.05, -0.025)
                # title = 'rzn_id = ' + str(i)
                # ax.set_title(title,fontsize = 22)
        if save_fig == True:        
            plt.savefig(join(self.prob_path, "coeffs"), bbox_inches = "tight", dpi = 300)
            plt.clf()
            plt.close()
        if show_fig == True:
            plt.show()


    def coeff_pair_plots(self,t, save_fig=True, show_fig=False):
        coeffs = self.all_Yi[t,:,0:4]
        df = pd.DataFrame(coeffs, columns = ['Coef. 1','Coef. 2','Coef. 3','Coef. 4'])
        sns.set_context("talk", rc={"font.size":10,"axes.titlesize":10,"axes.labelsize":25})   

        g= sns.pairplot(df, corner=True, diag_kind="kde", plot_kws=dict(s=20, edgecolor= 'b', linewidth=0.5, alpha=0.1))
        # g.map_lower(sns.kdeplot, levels=4, color=".2")
        # g.set(xlim=(-3, 3))
        # g.set(ylim=(-3, 3))
        for ax in g.axes[:,0]:
            ax.get_yaxis().set_label_coords(-0.31,0.5)
            # ax.set_xlim((-3,3))
            # ax.set_ylim((-3,3))

        fname = join(self.prob_path, "coeff_pairplots@") + str(t)
        if save_fig==True:
            plt.savefig(fname, dpi = 300)
            plt.clf()
            plt.close()
        if show_fig==True:
            plt.show()



prob_name = "DG3_g200x200x200_r5k_2LpDynObs_v2"
t = 1
rzn_id = 0
print("prob_name= ",prob_name)
prob_path = join(currentdir, prob_name)

plots = plot_environment_fields(prob_path)

rzn_list = [int(600)*i for i in range(8)] 
# plots.plot_coefs(rzn_list, save_fig=True)

# for t in [i for i in range(1,150,40)]:
for t in [1, 60, 120 , 180]:
    # plots.coeff_pair_plots(t, save_fig=True, show_fig=False)
    plots.plot_modes(t,save_fig=True, show_fig=False, show_colorbar=True)
#     plots.plot_vel_field(t, rzn_id, show_contours="vel_mag", show_colorbar=False, show_obstacle=False, save_fig=True, show_fig=False)
#     plots.plot_mean_vel_field(t, rzn_id, show_contours="vel_mag", show_colorbar=False, show_obstacle=False, save_fig=True, show_fig=False)

# # # plots.plot_modes(t,save_fig=True)

# # plots.plot_vel_field(t, rzn_id, show_contours="vel_mag", show_colorbar=True, show_obstacle=False, save_fig=True)
# # plots.plot_mean_vel_field(t, rzn_id, show_contours="vel_mag", show_colorbar=True, show_obstacle=False, save_fig=True)

# plots.plot_vel_field(t,rzn_id,show_obstacle=False, show_colorbar=True, save_fig=True)
# plots.plot_scalar_field(t,rzn_id, show_obstacle=True, show_colorbar=True, save_fig=True)
# plots.plot_scalar_field(t,rzn_id, show_obstacle=True, show_colorbar=True, save_fig=True)
# plots.plot_all(t,rzn_id, save_fig=True)
# plots.plot_obstacle(t)
# # plots.plot_env_gif()
# # plots.plot_env_sequence(rzn_id, plot_interval=30, show_contours="vel_mag")
plots.plot_env_sequence(rzn_id, plot_interval=60, show_contours="scalar_field")
# plots.plot_env_sequence(rzn_id, plot_interval=100, show_contours="vel_mag")
