import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
print(parentdir)


from build_env_only_scalar_fields import build_scalar_fields
from math import pi

init_gsize = 25
interpolate_degree = 4 # interpolates 
nt = 120
n_wsamples = 1000
n_modes = 4
op_nrzns = 1000 # used in plot_policy

dt = 20/nt
dxy = 2/init_gsize
final_gsize = init_gsize*interpolate_degree

A_sc = 3
eps = 0.1

w_range = ( pi/10, 8*pi/10 )
wx = pi
# wy = 0.5*pi   #hardcoded inside function

obs_upper_row = int(0.4*final_gsize)
obs_left_col = int(0.2*final_gsize)
obs_width = int(0.1*final_gsize)
obs_speed = [0.25, 0]
mode = 'multiple_dynamic'
dyn_obs_data = [obs_upper_row, obs_left_col, obs_width, obs_speed, mode]

build_scalar_fields(init_gsize, interpolate_degree, nt, dt, dxy, A_sc, eps,
                                                op_nrzns, n_wsamples, w_range, wx, 
                                                dyn_obstacle_data=dyn_obs_data)