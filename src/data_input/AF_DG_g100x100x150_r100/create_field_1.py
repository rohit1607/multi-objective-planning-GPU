import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
print(parentdir)


from build_environment_DG_with_westward_clouds import build_environment_DG_with_westward_clouds
from math import pi

init_gsize = 25
interpolate_degree = 4 # interpolates 
nt = 150
n_wsamples = 100
n_modes = 4
op_nrzns = 100 # used in plot_policy

dt = 1
dxy = 2/init_gsize  # fields built on grid of init_gsize and then interpolated
final_gsize = init_gsize*interpolate_degree

A = 0.5
A_sc = 3
eps = 0.1


w_range = ( pi/10, 8*pi/10 )
wx = pi
# wy = 0.5*pi   #hardcoded inside function

obs_upper_row = int(0.4*final_gsize)
obs_left_col = int(0.35*final_gsize)
obs_width = int(0.1*final_gsize)
obs_speed = [0.5, 0]
mode = 'static'
dyn_obs_data = [obs_upper_row, obs_left_col, obs_width, obs_speed, mode]

build_environment_DG_with_westward_clouds(init_gsize, nt, dt, dxy, A, A_sc, eps,
                                                op_nrzns, n_wsamples, w_range, wx, 
                                                interpolate_degree, n_modes, dyn_obstacle_data=dyn_obs_data)