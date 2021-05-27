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


    data = pandas.read_csv('./log.csv')
    number = int(sys.argv[1])
    print(number)
    print(range(number))

    arr_alpha = np.zeros(number+1)
    arr_time = np.zeros(number+1)
    arr_energy = np.zeros(number+1)
    
    for i in range(number+1):
        arr_alpha[i] = data['alpha_param'][len(data)-1-number+i]
        arr_time[i] = data['time_mean'][len(data)-1-number+i]
        arr_energy[i] = data['reqd_energy_mean'][len(data)-1-number+i]  
    
    plt.plot(arr_time, arr_energy, marker = 'o')
    plt.xlabel("Avg. Time Required")
    plt.ylabel("Avg. Energy Required")
    plt.title("Pareto-Optimal Front for Time-Energy")


    mypath = os.path.abspath(__file__)
    filepath = "data_solverOutput/" + prob_type + "/" + prob_name + "/" + prob_specs + "/"
    filename = "pareto_" + str(number) + ".png"
    plt.savefig(filepath + filename, bbox_inches = "tight", dp = 300)
    plt.clf()
    plt.close() 

    






