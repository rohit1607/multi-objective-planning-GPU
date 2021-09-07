import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import math
import imageio
import cv2
from pathlib import Path
import csv
import pandas
import sys
import os

if __name__ == "__main__":


    data = pandas.read_csv('./log.csv')
    logStart = int(sys.argv[1])
    logEnd = int(sys.argv[2])

    arr_build_time = np.zeros(logEnd-logStart+1)
    arr_spvi_time = np.zeros(logEnd-logStart+1)
    arr_plot_time = np.zeros(logEnd-logStart+1)
    arr_data_time = np.zeros(logEnd-logStart+1)
    arr_total_build_time = np.zeros(logEnd-logStart+1)
    #arr_data_time = np.zeros(logEnd-logStart+1)

    for i in np.linspace(logStart, logEnd, (logEnd-logStart+1), dtype=int):
        arr_build_time[i-logStart] = data['build_time_only'][i-2]
        arr_spvi_time[i-logStart] = data['spvi_time'][i-2]
        arr_plot_time[i-logStart] = data['data_time'][i-2]  
        arr_total_build_time[i-logStart] = data['build_time'][i-2]  
    
    BT_mean = np.mean(arr_build_time)
    BT_std = np.std(arr_build_time)
    ST_mean = np.mean(arr_spvi_time)
    ST_std = np.std(arr_spvi_time)
    PT_mean = np.mean(arr_plot_time)
    PT_std = np.std(arr_plot_time)
    TotalBT_mean = np.mean(arr_total_build_time)
    TotalBT_std = np.std(arr_total_build_time)

    #print(arr_build_time)

    print("Build Time")
    print("Mean: ", BT_mean, "s  ", "Std. Dev: ", BT_std, "s \n")

    print("Total Build Time")
    print("Mean: ", TotalBT_mean, "s  ", "Std. Dev: ", BT_std, "s \n")

    print("Solve Time")
    print("Mean: ", ST_mean, "s  ", "Std. Dev: ", ST_std, "s \n")

    print("Data Time")
    print("Mean: ", PT_mean, "s  ", "Std. Dev: ", PT_std, "s \n")



