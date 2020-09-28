import numpy as np


prob_name_1 = "src/data_modelOutput/time/test_g200x200x10_r10k/a1x4_i2_j7_ref1"
prob_name_2 = "src/data_modelOutput/time/test_g200x200x10_r10/a1x4_i2_j7_ref1"

s1file1 = prob_name_1 + "/master_cooS1.npy"
s1file2 = prob_name_2 + "/master_cooS1.npy"

s2file1 = prob_name_1 + "/master_cooS2.npy"
s2file2 = prob_name_2 + "/master_cooS2.npy"

valfile1 = prob_name_1 + "/master_cooVal.npy"
valfile2 = prob_name_2 + "/master_cooVal.npy"

R_file1 = prob_name_1 + "/master_R.npy"
R_file2 = prob_name_2 + "/master_R.npy"


S1_f1 = np.load(s1file1)
S1_f2 = np.load(s1file2)
S2_f1 = np.load(s2file1)
S2_f2 = np.load(s2file2)
V_f1 = np.load(valfile1)
V_f2 = np.load(valfile2)
R_f1 = np.load(R_file1)
R_f2 = np.load(R_file2)


def compare(np1, np2):
    print(len(np1) ,len(np2))
    assert(len(np1) == len(np2)),"array lenghts are not same!"
    status = True
    for i in range(len(np1)): 
        status = (np1[i][0] == np2[i][0])
        if status == False:
            print("Arrays NOT SAME. inequality encountered. Values are: ")
            print(i, np1[i][0] , np2[i][0])
            print()
            break
    if status == True:
        print("Arrays are SAME")
    return status




def verbose_compare(np1, np2, rng_st, rng_end, nprints=None):
    print(len(np1) ,len(np2))
    min_len = np.min([len(np1) ,len(np2)])
    # max_len = np.max([len(np1) ,len(np2)])
    if nprints == None:
        nprints = min_len

    for i in range(rng_st, rng_end):
        a = np1[i][0]
        b = np2[i][0]
        if a != b:
            status = "Diff"
        else:
            status = "same"
            
        # if status == "Diff":
        print(i, np1[i][0], "\t", np2[i][0], "\t", status)

        if i == nprints:
            break
    return True



def compare_coo_verbose(S1_f1, S1_f2, S2_f1, S2_f2, V_f1, V_f2, nprints = None):
    min_len = np.min([len(S1_f1) ,len(S1_f2)])
    # max_len = np.max([len(np1) ,len(np2)])
    if nprints == None:
        nprints = min_len

    for i in range(min_len):

        status = [0, 0, 0]

        if S1_f1[i][0] != S1_f2[i][0]:
            status[0] =  1
        if S2_f1[i][0] != S2_f2[i][0]:
            status[1] = 1
        if V_f1[i][0] != V_f2[i][0]:
            status[2] = 1

        if status == [0,0,0]:
            status = ""
            
        print(i, "\t| ", S1_f1[i][0], "\t", S2_f1[i][0], "\t", V_f1[i][0], "\t | \t",
                S1_f2[i][0], "\t", S2_f2[i][0], "\t", V_f2[i][0], "\t",  status)

        if i == nprints:
            break
    return True

def check_consecutive_pairs(S1_f1):
    l = len(S1_f1)
    count = 0
    for i in range(l-1):
        v1 =S1_f1[i][0]
        v2 =S1_f1[i+1][0]
        if v1 != v2 and v2 != v1+1:
            print(i,v1, v2, v2-v1)
            count+=1
    print("count = ", count)
    return

def test():
    print("----test 1----")
    a = np.arange(10).reshape(10,1)
    b = a
    print("arr1: ", a)
    print("arr2: ", b)
    compare(a, b)

    print("\n\n")
    print("----test 2----")
    c = np.arange(10).reshape(10,1)
    c[1][0] = 3
    print("arr1: ", a)
    print("arr2: ", c)
    compare(a,c)


def print_coo(S1, S2, V, num_prints):

    for i in range(num_prints):
         print(i, "\t| ", S1[i][0], "\t", S2[i][0], "\t", V[i][0])

# # test()
compare(S1_f1, S1_f2)
compare(S2_f1, S2_f2)
compare(V_f1, V_f2)
compare(R_f1, R_f2)

# verbose_compare(S1_f1, S1_f2, 22719, 23000)
# check_consecutive_pairs(S1_f2)