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
    logStart = int(sys.argv[1])
    logEnd = int(sys.argv[2])
    print(number)
    print(range(number))

    arr_alpha = np.zeros(logEnd-logStart+1)
    arr_time = np.zeros(logEnd-logStart+1)
    arr_energy = np.zeros(logEnd-logStart+1)
    arr_success = np.zeros(logEnd-logStart+1)
    yval = np.zeros(logEnd-logStart+1)
    


    for i in np.linspace(logStart, logEnd, (logEnd-logStart+1), dtype=int):
        arr_alpha[i-logStart] = data['alpha_param'][i-2]
        arr_time[i-logStart] = data['time_mean'][i-2]
        arr_energy[i-logStart] = data['net_energy_mean'][i-2]  
        arr_success[i-logStart] = data['success_rate'][i-2]
        prob_type = data['prob_type'][i-2]
        prob_name = data['prob_name'][i-2]
        prob_specs = data['prob_specs'][i-2]
    
    start = float(format(data['alpha_param'][logStart-2], ".3f"))
    end = float(format(data['alpha_param'][logEnd-2], ".3f"))
    delta = float(format((end-start)/(logEnd-logStart), ".3f"))

    cmap = plt.cm.tab20  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    #cmaplist.append((.5, .5, .5, 1.0))

    bounds = np.linspace(0, 1, 21)
    bounds2 = np.linspace(0, 2, 21)


    cmap = colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)

    fig, ax = plt.subplots()
    plt.rc('font', size=13)
    #ax_table.axis("off")
    #fig.subplots_adjust(right=1)

    yt = np.array([-100, -50, 0, 50])
    #xt = np.array([50, 75, 100])

    plt.yticks(yt, fontsize=20) 
    #plt.xticks(xt, fontsize=20)
    #plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)

    plt.scatter(arr_time, arr_energy, marker = 'o', c=arr_alpha, cmap=cmap)
    plt.plot(arr_time, arr_energy, 'k')
    plt.xlabel("Avg. Travel Time", fontsize=20)
    plt.ylabel("Avg. Net Energy", fontsize=20)
    #plt.title("Operating Curve for Time-Energy", fontsize=20)
    #cbar = plt.colorbar(cmap=cmap, orientation="horizontal", ticks=bounds2, boundaries=bounds, pad=0.2)
    #plt.xlim(27, 65)    #original limits
    #plt.ylim(40,135)    #original limits
    plt.xlim(30,105)
    plt.ylim(-120,90)
    #cbar.ax.tick_params(labelsize=15)


    cell_text = pandas.DataFrame({'alpha': arr_alpha.tolist(), 'success': arr_success.tolist()})
    #cell_text = pandas.DataFrame(arr_success, columns='success')
    #cell_text['alpha'] = arr_alpha.tolist()
    #cell_text['success'] = arr_success.tolist()

    #plt.table(cellText=cell_text.values, cellLoc='center' ,loc='right', colWidths=[0.1, 0.1])
    plt.tight_layout


    for i, txt in enumerate(arr_alpha):
        if(i!=0 and abs(arr_energy[i]-yval[i-1])<10):
            yval[i] = arr_energy[i] + 10
        else:
            yval[i] = arr_energy[i]
        #plt.annotate(txt, (arr_time[i], yval[i]))
    #for i, suc in enumerate(arr_success):
     #   plt.annotate(suc, (arr_time[i], arr_energy[i]+5))

    mypath = os.path.abspath(__file__)
    print(mypath)
    filepath = "data_solverOutput/" + prob_type + "/" + prob_name + "/" + prob_specs + "/"
    print(filepath)
    filename = "pareto_" + str(start) + "_" + str(delta) + "_" + str(end) + "_PAPER_final" + ".png"
    print(filename)
    plt.show()
    plt.savefig(filepath + filename, bbox_inches = "tight", dp = 300)
    plt.clf()
    plt.close() 

    






