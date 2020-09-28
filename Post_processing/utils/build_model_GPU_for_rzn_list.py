import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from math import pi, sin, cos, atan2
import time
from collections import Counter
from multiprocessing import Pool
import pickle
from utils.setup_grid import setup_grid
# from utils.build_model import write_files
from definition import ROOT_DIR
from os.path import join, exists
from utils.custom_functions import createFolder, append_summary_to_summaryFile, read_pickled_File
from os import getcwd, makedirs


def pickleFile(file, filename):
    with open(filename + '.p', 'wb') as fp:
        pickle.dump(file, fp, protocol=pickle.HIGHEST_PROTOCOL)

def write_files(transition_dict, filename, data):
    """
    Pickles dictionary contaniing model details.
    Writes parameters to file.
    Writes parameters to summary file
    :param transition_dict:
    :param filename:
    :return:
    """

    summary_file = base_path + 'model_summary.csv'
    params, param_str, reward_structure, build_time = data

    createFolder(save_path)

    # save transition_probs. Pickle it.
    with open(save_path + '/' + filename + '.p', 'wb') as fp:
        pickle.dump(transition_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open(save_path + '/' + filename + '_params.txt', 'w') as f:
        for i in range(len(param_str)):
            f.write(param_str[i] + ':' + '    ' + str(params[i]) + "\n")
        f.write("Reward Structure: " + str(reward_structure) + "\n")
        f.write("Build Time: "+ str(build_time))

    # append_summary_to_summaryFile(summary_file, )
#transition_dict is a nested dictionary
def initialise_dict(g):
    transition_dict = {}
    for s in state_list:
        transition_dict[s]={}
        for a in g.actions:
            transition_dict[s][a]={}

    return transition_dict


def get_S_from_S_id(S_id, gsize):
    """
    returns (i,j) from combined state ID. this is for 2d
    :param S_id: combined id of state- one number encapsulating t,i,j  or i,j
    :param gsize: dim of grid in 1direction
    :return: S in t
    TODO: Add t for 3d case
    """
    t = int(S_id // (gsize**2))
    rem_t = int(S_id % (gsize**2))
    # if S_id >= gsize**2:
    #     print("prob: ",S_id)
    i = int(rem_t // gsize) 
    j = int(rem_t % gsize)
    S = (t,i,j)

    return S


def convert_COO_to_dict(tdict, g, coo_list, Rs_list):
    """
    takes in saved coo and Rsa files. unpacks them and converts them to dict
    :param init_transition_dict:
    :param g:
    :param coo_file:
    :param Rsa_file:
    :return:
    TODO: add another loop for T for 3d case
    """
    # coo_list = read_pickled_File(coo_file)
    # Rs_list = read_pickled_File(Rsa_file)
    num_actions = len(g.actions)

    for i in range(num_actions):
        coo = coo_list[i]
        Rs = Rs_list[i]
        m, n = coo.shape
        a = g.actions[i]
        for k in range(n):
            S1 = get_S_from_S_id(coo[0, k], g.ni)
            S2 = get_S_from_S_id(coo[1, k], g.ni)
            prob = coo[2,k]
            r = Rs[int(coo[0,k])]
            try:
                tdict[S1][a][S2] = (prob, r)
            except:
                # pass
                if S1[0] == 0:
                    print(S1, 'is not an actionable state', a, S2, prob, r)

    return tdict


def get_COO_(results, nrzns, T, save_filename=None):
    """
    results is a 1D array of size ncells*nrzns with each element being an S2(ij)
    This functions performs a reduction and counts no. of S2(t'ij) each S1(tij)
    the conversion from S1(ij) to S1(tij) and  here explicitly. S2 conversion is already done outside the function
    Returns sparse matrix in coo row major format where the rows are S1(tij), S2(t'i'j'), prob
    TODO:xxDONE 1. get s1 and s2 as f(T)
         xxDONE 2. sort to row/col major format
    """
    ncells = int(len(results) / nrzns)  # no. of spatial cells in grid
    counter_list = []  # to store a counter() for each s1
    for i in range(ncells):
        counter_list.append(Counter())

    # increment counter for each cell
    for i in range(ncells):
        for j in results[i::ncells]:  # strides of ncells will s2 info about same s1
            counter_list[i][j] += 1

    # print("test/n",counter_list[0])
    nnz = 0  # count no. of nnz
    for i in range(ncells):
        nnz += len(counter_list[i])

    coo = np.zeros((3, nnz))
    cnt = 0
    for i in range(ncells):
        # get s1 as function of T
        s1 = int(i + T * (ncells))
        for s2 in sorted(counter_list[i]):
            coo[:, cnt] = [s1, s2, counter_list[i][s2] / nrzns]
            cnt += 1

    if save_filename != None:
        np.save(save_filename, coo)

    return coo


def concatenate_results_across_time(coo_list_a, Rs_list_a, full_coo_list_a, full_Rs_list_a):
    num_actions = len(coo_list_a)
    for i in range(num_actions):
        # concatenate along columns
        full_coo_list_a[i] = np.concatenate((full_coo_list_a[i], coo_list_a[i]), axis=1)
        # concatenate along row
        full_Rs_list_a[i] = np.concatenate((full_Rs_list_a[i], Rs_list_a[i]), axis=0)

    return full_coo_list_a, full_Rs_list_a


def build_sparse_transition_model_at_T(T, T_gpu, vel_data_gpu, params, bDimx, params_gpu, xs_gpu, ys_gpu, ac_angles,
                                       results, sumR_sa, save_file_for_each_a=False):
    gsize = int(params[0])
    num_actions = int(params[1])
    nrzns = int(params[2])

    all_u_mat_gpu, all_v_mat_gpu, all_ui_mat_gpu, all_vi_mat_gpu, all_Yi_gpu = vel_data_gpu

    results_gpu_list = []
    sumR_sa_gpu_list = []
    for i in range(num_actions):
        results_gpu_list.append(cuda.mem_alloc(results.nbytes))
        sumR_sa_gpu_list.append(cuda.mem_alloc(sumR_sa.nbytes))
    for i in range(num_actions):
        cuda.memcpy_htod(results_gpu_list[i], results)
        cuda.memcpy_htod(sumR_sa_gpu_list[i], sumR_sa)

    print("alloted mem in inner func")


    # let one thread access a state centre. access coresponding velocities, run all actions
    # TODO: dt may not be int for a genral purpose code

    mod = SourceModule("""
    __device__ int32_t get_thread_idx()
            // assigns idx to thread with which it accesses the flattened 3d vxrzns matrix
            // for a given T and a given action. 
            // runs for both 2d and 3d grid
            // TODO: may have to change this considering cache locality
        {
            // here i, j, k refer to a general matrix M[i][j][k]
            int32_t i = threadIdx.x;
            int32_t j = blockIdx.y;
            int32_t k = blockIdx.x;
            int32_t idx = k + (j*gridDim.x)  + (i*gridDim.x*gridDim.y)+ blockIdx.z*blockDim.x*gridDim.x*gridDim.y;
            return idx;
        }
    __device__ int32_t state1D_from_thread(int32_t T)
    {   
        // j ~ blockIdx.x
        // i ~ blockIdx.y 
        // The above three consitute a spatial state index from i and j of grid
        // last term is for including time index as well.
        return (blockIdx.x + (blockIdx.y*gridDim.x) + (T*gridDim.x*gridDim.y) ); 
    }
    __device__ int32_t state1D_from_ij(int32_t*  posid, int32_t T)
    {
        // posid = {i , j}
        // state id = j + i*dim(i) + T*dim(i)*dim(j)
        return (posid[1] + posid[0]*gridDim.x + (T*gridDim.x*gridDim.y) ) ; 
    }
    __device__ bool is_edge_state(int32_t i, int32_t j)
    {
        // n = gsize -1 that is the last index of the domain assuming square domain
        int32_t n = gridDim.x - 1;
        if (i == 0 || i == n || j == 0 || j == n ) 
            {
                return true;
            }
        else return false;
    }
    __device__ bool is_terminal(int32_t i, int32_t j, float* params)
    {
        int32_t i_term = params[8];         // terminal state indices
        int32_t j_term = params[9];
        if(i == i_term && j == j_term)
        {
            return true;
        }
        else return false;
    }
    __device__ bool my_isnan(int s)
    {
    // By IEEE 754 rule, NaN is not equal to NaN
    return s != s;
    }
    __device__ void get_xypos_from_ij(int32_t i, int32_t j, float* xs, float* ys, float* x, float* y)
    {
        *x = xs[j];
        *y = ys[gridDim.x - 1 - i];
        return;
    }
    __device__ float get_angle_in_0_2pi(float theta)
    {
        float f_pi = 3.141592;
        if (theta < 0)
        {
            return theta + (2*f_pi);
        }
        else
        {
            return theta;
        }  
    }
    __device__ float calculate_reward_const_dt(float* xs, float* ys, int32_t i_old, int32_t j_old, float xold, float yold, int32_t* newposids, float* params, float vnet_x, float vnet_y )
    {
        // xold and yold are centre of old state (i_old, j_old)
        float dt = params[4];
        float r1, r2, theta1, theta2, theta, h;
        float dt_new;
        float xnew, ynew;
        if (newposids[0] == i_old && newposids[1] == j_old)
        {
            dt_new = dt;
        }
        else
        {
            get_xypos_from_ij(newposids[0], newposids[1], xs, ys, &xnew, &ynew); //get centre of new states
            h = sqrtf((xnew - xold)*(xnew - xold) + (ynew - yold)*(ynew - yold));
            r1 = h/(sqrtf((vnet_x*vnet_x) + (vnet_y*vnet_y)));
            theta1 = get_angle_in_0_2pi(atan2f(vnet_y, vnet_x));
            theta2 = get_angle_in_0_2pi(atan2f(ynew - yold, xnew - xold));
            theta = fabsf(theta1 -theta2);
            r2 = fabsf(sinf(theta));
            dt_new = r1 + r2;
            if (threadIdx.x == 0 && blockIdx.z == 0 && blockIdx.x == 1 && blockIdx.y == 1)
            {
                params[24] = r1;
                params[25] = r2;
            }
        }
        return -dt_new;
    }
    __device__ void move(float ac_angle, float vx, float vy, float* xs, float* ys, int32_t* posids, float* params, float* r )
    {
            int32_t n = params[0] - 1;      // gsize - 1
            // int32_t num_actions = params[1];
            // int32_t nrzns = params[2];
            float F = params[3];
            float dt = params[4];
            float r_outbound = params[5];
            float r_terminal = params[6];
            float Dj = fabsf(xs[1] - xs[0]);
            float Di = fabsf(ys[1] - ys[0]);
            float r_step = 0;
            *r = 0;
            int32_t i0 = posids[0];
            int32_t j0 = posids[1];
            float vnetx = F*cosf(ac_angle) + vx;
            float vnety = F*sinf(ac_angle) + vy;
            float x, y;
            get_xypos_from_ij(i0, j0, xs, ys, &x, &y); // x, y stores centre coords of state i0,j0
            float xnew = x + (vnetx * dt);
            float ynew = y + (vnety * dt);
            //checks TODO: remove checks once verified
            if (threadIdx.x == 0 && blockIdx.z == 0 && blockIdx.x == 1 && blockIdx.y == 1)
            {
                params[12] = x;
                params[13] = y;
                params[14] = vnetx;
                params[15] = vnety;
                params[16] = xnew;
                params[17] = ynew;
                params[18] = ac_angle;
            }
            if (xnew > xs[n])
                {
                    xnew = xs[n];
                    *r += r_outbound;
                }
            else if (xnew < xs[0])
                {
                    xnew = xs[0];
                    *r += r_outbound;
                }
            if (ynew > ys[n])
                {
                    ynew =  ys[n];
                    *r += r_outbound;
                }
            else if (ynew < ys[0])
                {
                    ynew =  ys[0];
                    *r += r_outbound;
                }
            // TODO:xxDONE check logic wrt remainderf. remquof had issue
            int32_t xind, yind;
            //float remx = remquof((xnew - xs[0]), Dj, &xind);
            //float remy = remquof(-(ynew - ys[n]), Di, &yind);
            float remx = remainderf((xnew - xs[0]), Dj);
            float remy = remainderf(-(ynew - ys[n]), Di);
            xind = ((xnew - xs[0]) - remx)/Dj;
            yind = (-(ynew - ys[n]) - remy)/Di;
            if ((remx >= 0.5 * Dj) && (remy >= 0.5 * Di))
                {
                    xind += 1;
                    yind += 1;
                }
            else if ((remx >= 0.5 * Dj && remy < 0.5 * Di))
                {
                    xind += 1;
                }
            else if ((remx < 0.5 * Dj && remy >= 0.5 * Di))
                {
                    yind += 1;
                }
            if (!(my_isnan(xind) || my_isnan(yind)))
                {
                    posids[0] = yind;
                    posids[1] = xind;
                    if (is_edge_state(posids[0], posids[1]))     //line 110
                        {
                            *r += r_outbound;
                        }
                    
                    if (threadIdx.x == 0 && blockIdx.z == 0 && blockIdx.x == 1 && blockIdx.y == 1)
                    {
                        params[26] = 9999;
                    }
                }
            r_step = calculate_reward_const_dt(xs, ys, i0, j0, x, y, posids, params, vnetx, vnety);
            //TODO: change back to normal when needed
            //r_step = -dt;
            *r += r_step; //TODO: numerical check remaining
            if (is_terminal(posids[0], posids[1], params))
                {
                    *r += r_terminal;
                }
            
            if (threadIdx.x == 0 && blockIdx.z == 0 && blockIdx.x == 1 && blockIdx.y == 1)
            {
                params[19] = xnew;
                params[20] = ynew;
                params[21] = yind;
                params[22] = xind;
                params[23] = *r;
                //params[17] = ynew;
                //params[18] = ac_angle;
            }
    }


    __device__ void extract_velocity(float* vx, float* vy, int32_t T, float* all_u_mat, float* all_v_mat, float* all_ui_mat, float* all_vi_mat, float* all_Yi, float* params)
    {
        int32_t nrzns = params[2];
        int32_t nmodes = params[7];              
        
        int32_t sp_uvi, str_uvi, sp_Yi, str_Yi; //startpoints and strides for accessing all_ui_mat, all_vi_mat and all_Yi
        float sum_x = 0;
        float sum_y = 0;
        float vx_mean, vy_mean;

        //thread index. also used to access resultant vxrzns[nrzns, gsize, gsize]
        int32_t idx = get_thread_idx();

        //rzn index to identify which of the 5k rzn it is. used to access all_Yi.
        int32_t rzn_id = (blockIdx.z * blockDim.x)  + threadIdx.x ;

        //mean_id is the index used to access the flattened all_u_mat[t,i,j].
        int32_t mean_id = state1D_from_thread(T);

        //to access all_ui_mat and all_vi_mat
        str_uvi = gridDim.x * gridDim.y;
        sp_uvi = (T * nmodes * str_uvi) + (gridDim.x * blockIdx.y) + (blockIdx.x);

        // to access all_Yi
        sp_Yi = (T * nrzns * nmodes) + (rzn_id * nmodes);

        vx_mean = all_u_mat[mean_id];
        for(int i = 0; i < nmodes; i++)
        {
            sum_x += all_ui_mat[sp_uvi + (i*str_uvi)]*all_Yi[sp_Yi + i];
        }

        vy_mean = all_v_mat[mean_id];
        for(int i = 0; i < nmodes; i++)
        {
            sum_y += all_vi_mat[sp_uvi + (i*str_uvi)]*all_Yi[sp_Yi + i];
        }
        
        *vx = vx_mean + sum_x;
        *vy = vy_mean + sum_y;
     
        return;
    }


    //test: changer from float* to float ac_angle
    __global__ void transition_calc(float* T_arr, float* all_u_mat, float* all_v_mat, float* all_ui_mat, float* all_vi_mat, float* all_Yi,
                                    float ac_angle, float* xs, float* ys, float* params, float* sumR_sa, float* results)
                                            // resutls directions- 1: along S2;  2: along S1;    3: along columns towards count
    {
        int32_t gsize = params[0];          // size of grid along 1 direction. ASSUMING square grid.
        int32_t num_actions = params[1];    
        int32_t nrzns = params[2];
        float F = params[3];
        float dt = params[4];
        float r_outbound = params[5];
        float r_terminal = params[6];
        int32_t nmodes = params[7];              
        int32_t i_term = params[8];         // terminal state indices
        int32_t j_term = params[9];
        int32_t nT = params[10];
        int32_t is_stationary = params[11];
        int32_t T = (int32_t)T_arr[0];
        int32_t idx = get_thread_idx();

        float vx, vy;

        if(idx < gridDim.x*gridDim.y*nrzns)
        {
            int32_t posids[2] = {blockIdx.y, blockIdx.x};    //static declaration of array of size 2 to hold i and j values of S1. 
            int32_t sp_id;      //sp_id is space_id. S1%(gsize*gsize)

            //  Afer move() these will be overwritten by i and j values of S2
            float r;              // to store immediate reward

            extract_velocity(&vx, &vy, T, all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, all_Yi, params);

            //move(*ac_angle, vx, vy, xs, ys, posids, params, &r);
            move(ac_angle, vx, vy, xs, ys, posids, params, &r);
            int32_t S1, S2;
            if (is_stationary == 1)
            {
                T = 0;
                S1 = state1D_from_thread(T);     //get init state number corresponding to thread id
                S2 = state1D_from_ij(posids, T);   //get successor state number corresponding to posid and next timestep T+1        
            }
            else
            {
                S1 = state1D_from_thread(T);     //get init state number corresponding to thread id
                S2 = state1D_from_ij(posids, T+1);   //get successor state number corresponding to posid and next timestep T+1        
                sp_id = S1%(gsize*gsize);
            }
            //writing to sumR_sa. this array will later be divided by num_rzns, to get the avg
            float a = atomicAdd(&sumR_sa[sp_id], r); //TODO: try reduction if this is slow overall
            results[idx] = S2;
            __syncthreads();
            /*if (threadIdx.x == 0 && blockIdx.z == 0)
            {
                sumR_sa[S1] = sumR_sa[S1]/nrzns;    //TODO: change name to R_sa from sumR_sa since were not storing sum anymore
            }
           */
        }//if ends
        return;
    }
        """)

    # sumR_sa2 = np.empty_like(sumR_sa, dtype = np.float32)
    # cuda.memcpy_dtoh(sumR_sa2, sumR_sa_gpu)
    # print("sumR_sa",sumR_sa)
    # print("sumR_sa",sumR_sa2[0:10001])
    # T = np.array(T64, dtype = np.float32)
    params2 = np.empty_like(params).astype(np.float32)
    func = mod.get_function("transition_calc")
    for i in range(num_actions):
        print('T', T, " call kernel for action: ",i)
        func(T_gpu, all_u_mat_gpu, all_v_mat_gpu, all_ui_mat_gpu, all_vi_mat_gpu, all_Yi_gpu, ac_angles[i], xs_gpu, ys_gpu, params_gpu, sumR_sa_gpu_list[i], results_gpu_list[i],
             block=(bDimx, 1, 1), grid=(gsize, gsize, (nrzns // bDimx) + 1))
        if i == 0:
            cuda.memcpy_dtoh(params2, params_gpu)
            print("params check:",)
            print(  '\nangle= ', params2[18],
                    '\nx =' ,params2[12],
                '\ny =' ,params2[13] ,
                    '\nvnetx = ',params2[14],
                    '\nvnety =', params2[15],
                    '\nxnew =', params2[16],
                    '\nynew =', params2[17],
                    '\nxnewupd =', params2[19],
                    '\nynewupd =', params2[20],
                    '\nyind i=', params2[21],
                    '\nxind j=', params2[22],
                    '\nr- =', params2[23],
                    '\nr1+ =', params2[24],
                    '\nr2+ =', params2[25],
                    '\nenter_isnan =', params2[26]
                )

    results2_list = []
    sum_Rsa2_list = []
    for i in range(num_actions):
        results2_list.append(np.empty_like(results))
        sum_Rsa2_list.append(np.empty_like(sumR_sa))

    # SYNCHRONISATION - pycuda does it implicitly.

    for i in range(num_actions):
        cuda.memcpy_dtoh(results2_list[i], results_gpu_list[i])
        cuda.memcpy_dtoh(sum_Rsa2_list[i], sumR_sa_gpu_list[i])
        print("memcpy_dtoh for action: ", i)


    for i in range(num_actions):
        sum_Rsa2_list[i] = sum_Rsa2_list[i] / nrzns

    # print("sumR_sa2\n",sumR_sa2,"\n\n")

    # print("results_a0\n",results2_list[0].T[50::int(gsize**2)])
    print("OK REACHED END OF cuda relevant CODE\n")

    # make a list of inputs, each elelment for an action. and run parallal get_coo_ for each action
    # if save_file_for_each_a is true then each file must be named appopriately.
    if save_file_for_each_a == True:
        f1 = 'COO_Highway2D_T' + str(T) + '_a'
        f3 = '_of_' + str(num_actions) + 'A.npy'
        inputs = [(results2_list[i], nrzns, T, f1 + str(i) + f3) for i in range(num_actions)]
    else:
        inputs = [(results2_list[i], nrzns, T, None) for i in range(num_actions)]

    # coo_list_a is a list of coo for each each action for the given timestep.
    with Pool(num_actions) as p:
        coo_list_a = p.starmap(get_COO_, inputs)
    # print("coo print\n", coo.T[4880:4900, :])
    print("\n\n")
    # print("time taken by cuda compute and transfer\n", (t2 - t1) / 60)
    # print("time taken for post processing to coo on cpu\n",(t3 - t2) / 60)

    return coo_list_a, sum_Rsa2_list


# def build_sparse_transition_model(filename, num_actions, nt, dt, F, end_ij):
def build_sparse_transition_model(filename = 'Transition_dict', n_actions = 16, all_Yi = None, wanted_nrzns =5000, nt = None):
    
    global state_list
    global base_path
    global save_path

    print("Building Sparse Model")
    t1 = time.time()
    #setup grid
    print("input to build_sparse_trans_model:\n")
    print("n_actions", n_actions)
    # print("nt, dt", nt, dt)

    g, xs, ys, X, Y, vel_field_data, nmodes, num_rzns, path_mat, setup_params, setup_param_str = setup_grid(num_actions = n_actions)

    print("xs: ",xs)
    print("ys", ys)

    # ------- IMPORTANT ----------
    # REMOVING all_Yi FROM ROLLING OUT vel_field_data BECAUSE WE WILL USE A REPLACEMENT
    all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, _ = vel_field_data


    check_nt, check_nrzns, nmodes = all_Yi.shape

    all_u_mat = all_u_mat.astype(np.float32)
    all_v_mat = all_v_mat.astype(np.float32)
    all_ui_mat = all_ui_mat.astype(np.float32)
    all_vi_mat = all_vi_mat.astype(np.float32)
    all_Yi = all_Yi.astype(np.float32)


    #setup_params = [num_actions, nt, dt, F, startpos, endpos] reference from setup grid
    nT = setup_params[1]  # total no. of time steps TODO: check default value
    print("****CHECK: ", nt, nT, check_nt)
    # assert (nt == nT), "nt and nT are not the same!"
    #if nt specified in runner is within nT from param file, then use nt. i.e. model will be built for nt timesteps.
    if nt != None and nt <= nT:
        nT = nt
    is_stationary = 0  # 0 is false. any other number is true. is_stationry = 0 (false) means that flow is NOT stationary
    #  and S2 will be indexed by T+1. if is_stationary = x (true), then S2 is indexed by 0, same as S1.
    # list_size = 10     #predefined size of list for each S2
    # if nt > 1:
    #     is_stationary = 0
    gsize = g.ni  # size of grid along 1 direction. ASSUMING square grid.
    num_actions = setup_params[0]

    # nrzns = num_rzns
    nrzns = wanted_nrzns

    bDimx = nrzns # for small test cases
    if nrzns>=1000:
        bDimx = 1000   #for large problems     
    dt = setup_params[2]
    F = setup_params[3]
    r_outbound = g.r_outbound
    r_terminal = g.r_terminal
    i_term = g.endpos[0]  # terminal state indices
    j_term = g.endpos[1]

    #name of output pickle file containing transtion prob in dictionary format
    if nT > 1:
        prefix = '3D_' + str(nT) + 'nT_a'
    else:
        prefix = '2D_a'
    filename =  filename + prefix + str(n_actions) #TODO: change filename
    base_path = join(ROOT_DIR,'DP/Trans_matxs_3D/')
    save_path = base_path + filename
    if exists(save_path):
        print("Folder Already Exists !!\n")
        return
    # TODO: remove z from params. it is only for chekcs
    z=-9999
    params = np.array(
        [gsize, num_actions, nrzns, F, dt, r_outbound, r_terminal, nmodes, i_term, j_term, nT, is_stationary, z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z]).astype(
        np.float32)
    st_sp_size = (gsize ** 2) # size of spatial state space
    print("check stsp_size", gsize, nT, st_sp_size)
    save_file_for_each_a = False

    print("params")
    print("gsize ", params[0], "\n",
        "num_actions ", params[1], "\n",
        "nrzns ", params[2], "\n",
        "F ", params[3], "\n",
        "dt ", params[4], "\n",
        "r_outbound ", params[5], "\n",
        "r_terminal ", params[6], "\n",
        "nmodes ", params[7], "\n",
        "i_term ", params[8], "\n",
        "j_term ", params[9], "\n",
        "nT", params[10], "\n",
        "is_stationary ", params[11], ""
    
        )


    # cpu initialisations.
    # dummy intialisations to copy size to gpu
    # vxrzns = np.zeros((nrzns, gsize, gsize), dtype=np.float32)
    # vyrzns = np.zeros((nrzns, gsize, gsize), dtype=np.float32)

    results = -1 * np.ones(((gsize ** 2) * nrzns), dtype=np.float32)
    sumR_sa = np.zeros(st_sp_size).astype(np.float32)
    Tdummy = np.zeros(2, dtype = np.float32)

    #  informational initialisations
    ac_angles = np.linspace(0, 2 * pi, num_actions, endpoint =  False, dtype=np.float32)
    print("action angles:\n", ac_angles)

    ac_angle = ac_angles[0].astype(np.float32) # just for allocating memory
    # xs = np.arange(gsize, dtype=np.float32)
    # ys = np.arange(gsize, dtype=np.float32)
    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)
    print("params: \n", params, "\n\n")

    t1 = time.time()
    # allocates memory on gpu. vxrzns and vyrzns nees be allocated just once and will be overwritten for each timestep
    # vxrzns_gpu = cuda.mem_alloc(vxrzns.nbytes)
    # vyrzns_gpu = cuda.mem_alloc(vyrzns.nbytes)
    all_u_mat_gpu = cuda.mem_alloc(all_u_mat.nbytes)
    all_v_mat_gpu = cuda.mem_alloc(all_v_mat.nbytes)
    all_ui_mat_gpu = cuda.mem_alloc(all_ui_mat.nbytes)
    all_vi_mat_gpu = cuda.mem_alloc(all_vi_mat.nbytes)
    all_Yi_gpu = cuda.mem_alloc(all_Yi.nbytes)    
    vel_data_gpu = [all_u_mat_gpu, all_v_mat_gpu, all_ui_mat_gpu, all_vi_mat_gpu, all_Yi_gpu]

    ac_angles_gpu = cuda.mem_alloc(ac_angles.nbytes)
    ac_angle_gpu = cuda.mem_alloc(ac_angle.nbytes)
    xs_gpu = cuda.mem_alloc(xs.nbytes)
    ys_gpu = cuda.mem_alloc(ys.nbytes)
    params_gpu = cuda.mem_alloc(params.nbytes)
    T_gpu = cuda.mem_alloc(Tdummy.nbytes)


    # copies contents of a to  allocated memory on gpu
    cuda.memcpy_htod(all_u_mat_gpu, all_u_mat)
    cuda.memcpy_htod(all_v_mat_gpu, all_v_mat)
    cuda.memcpy_htod(all_ui_mat_gpu, all_ui_mat)
    cuda.memcpy_htod(all_vi_mat_gpu, all_vi_mat)
    cuda.memcpy_htod(all_Yi_gpu, all_Yi)

    cuda.memcpy_htod(ac_angle_gpu, ac_angle)
    cuda.memcpy_htod(xs_gpu, xs)
    cuda.memcpy_htod(ys_gpu, ys)
    cuda.memcpy_htod(params_gpu, params)

    for T in range(nT):
        print("*** Computing data for timestep, T = ", T, '\n')
        # params[7] = T
        # cuda.memcpy_htod(params_gpu, params)
        Tdummy[0] = T
        # Load Velocities
        # vxrzns = np.zeros((nrzns, gsize, gsize), dtype = np.float32)
        # #expectinf to see probs of 0.5 in stream area
        # for i in range(int(nrzns/2)):
        #     vxrzns[i,int(gsize/2 -1):int(gsize/2 +1),:] = 1
        # vyrzns = np.zeros((nrzns, gsize, gsize), dtype = np.float32)
        # vxrzns = np.load('/home/rohit/Documents/Research/ICRA_2020/DDDAS_2D_Highway/Input_data_files/Velx_5K_rlzns.npy')
        # vyrzns = np.load('/home/rohit/Documents/Research/ICRA_2020/DDDAS_2D_Highway/Input_data_files/Vely_5K_rlzns.npy')
        # vxrzns = Vx_rzns
        # vyrzns = Vy_rzns
        # vxrzns = vxrzns.astype(np.float32)
        # vyrzns = vyrzns.astype(np.float32)
        Tdummy = Tdummy.astype(np.float32)

        # TODO: sanity check on dimensions: compare loaded matrix shape with gsize, numrzns

        # copy loaded velocities to gpu
        # cuda.memcpy_htod(vxrzns_gpu, vxrzns)
        # cuda.memcpy_htod(vyrzns_gpu, vyrzns)
        cuda.memcpy_htod(T_gpu, Tdummy)

        print("pre func")

        coo_list_a, Rs_list_a = build_sparse_transition_model_at_T(T, T_gpu, vel_data_gpu, params, bDimx, params_gpu,
                                                                   xs_gpu, ys_gpu,
                                                                   ac_angles, results, sumR_sa,
                                                                   save_file_for_each_a=False)

        # print("R_s_a0 \n", Rs_list_a[0][0:200])
        print("post func")


        # TODO: end loop over timesteps here and comcatenate COOs and R_sas over timesteps for each action
        # full_coo_list and full_Rs_list are lists with each element containing coo and R_s for an action of the same index
        if T > 0:
            full_coo_list_a, full_Rs_list_a = concatenate_results_across_time(coo_list_a, Rs_list_a, full_coo_list_a,
                                                                              full_Rs_list_a)
            # TODO: finish concatenate...() function
        else:
            full_coo_list_a = coo_list_a
            full_Rs_list_a = Rs_list_a

    t2 = time.time()
    build_time = t2 - t1
    print("build_time ", build_time)

    #save data to file
    # data = setup_params, setup_param_str, g.reward_structure, build_time
    # write_files(full_coo_list_a, filename + '_COO', data)

    # print("Pickled sparse files !")

    #build probability transition dictionary
    state_list = g.ac_state_space()
    init_transition_dict = initialise_dict(g)
    transition_dict = convert_COO_to_dict(init_transition_dict, g, full_coo_list_a, full_Rs_list_a)
    print("conversion COO to dict done")

    #save dictionary to file
    data = setup_params, setup_param_str, g.reward_structure, build_time
    write_files(transition_dict, filename, data)
    pickleFile(full_coo_list_a, save_path + '/' + filename + '_COO')
    pickleFile(full_Rs_list_a, save_path + '/' + filename + '_Rsa')