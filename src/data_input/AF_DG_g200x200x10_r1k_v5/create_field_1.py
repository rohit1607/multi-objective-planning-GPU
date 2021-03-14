import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
print(parentdir)


from build_environment_DG_with_westward_clouds import build_environment_DG_with_westward_clouds
from math import pi

init_gsize = 25
nt = 10
dt = 10/nt
dxy = 2/init_gsize
A = 0.5
A_sc = 3
eps = 0.1
op_nrzns = 100
n_wsamples = 1000
w_range = ( pi/10, 8*pi/10 )
# wy = 0.5*pi   #hardcoded inside function
wx = pi
# interpolates 
interpolate_degree = 8
n_modes = 4

build_environment_DG_with_westward_clouds(init_gsize, nt, dt, dxy, A, A_sc, eps,
                                                op_nrzns, n_wsamples, w_range, wx, 
                                                interpolate_degree, n_modes)