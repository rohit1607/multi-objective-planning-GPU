import numpy as np
from os.path import join

"""
To convert long long int coos from build_model.cu to int format
"""

# file contains path to where modelOutput to be converted is stored
file = open(r"temp_modelOp_dirName.txt","r") 

# read lines as string from file line by line
file_lines = file.readlines()

# 4th line contains address of model output
prob_name = file_lines[3]
print("read line:\n", prob_name)


master_cooS1 = np.load(join(prob_name, 'master_cooS1.npy')).astype(np.int32)
master_cooS2 = np.load(join(prob_name, 'master_cooS2.npy')).astype(np.int32)

np.save(join(prob_name, 'master_cooS1.npy'), master_cooS1)
np.save(join(prob_name, 'master_cooS2.npy'), master_cooS2)

print("Conversion Finished")


