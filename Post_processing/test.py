import numpy as np
import math
from plot_e2eGPU_policy import bilinear_interpolate, traingle_interpolate
import matplotlib.pyplot as plt

# def bilinear_interpolate(x, y, x_tuple, y_tuple, ordered_vals):

# def traingle_interpolate(x, y, p_list, v_list):


def bil_interp_test1(xy_i, xy_f):
    x_tuple = (0, 2)
    y_tuple = (0, 2)

    v11 = 1
    v12 = 2
    v21 = 3
    v22 = 4
    ordered_vals = [v11, v12, v21, v22]

    xi, yi = xy_i
    xf, yf = xy_f
    n = 10
    xy_array = np.linspace(xy_i, xy_f, n)
    vals =  np.zeros((n,))
    for i in range(n):
        x, y = xy_array[i,:]
        vals[i] = bilinear_interpolate(x,y,x_tuple, y_tuple, ordered_vals)
        
    plt.plot(vals)
    plt.savefig('test_bilinear_interpolate'+ str(xy_i)+str(xy_f))
    plt.cla()

def triangle_interp_test(xy_i, xy_f):

    x_tuple = (0, 2)
    y_tuple = (0, 2)
    x1, x2 = x_tuple
    y1, y2 = y_tuple

    v11 = 1
    v12 = 2
    v21 = 3
    v22 = 4

    p_list = [(x1, y1), (x1, y2), (x2, y1)]
    v_list = [v11, v12, v21]

    n = 10
    xy_array = np.linspace(xy_i, xy_f, n)
    vals =  np.zeros((n,))
    for i in range(n):
        x, y = xy_array[i,:]
        vals[i] = traingle_interpolate(x, y, p_list, v_list)
        
    plt.plot(vals)
    plt.savefig('test_triangle_interpolate'+ str(xy_i)+str(xy_f))
    plt.cla()

# print(v)
# print(math.atan2(3**0.5,1) * 180/math.pi)
bil_interp_test1((1,0), (1,2))
bil_interp_test1((0,1), (2,1))
bil_interp_test1((0,0), (2,2))
bil_interp_test1((0,0), (2,1))

triangle_interp_test((1,0),(0,2))
triangle_interp_test((0,0),(1,1))






