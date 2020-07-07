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





def build_sparse_transition_model(filename = 'Transition_dict', n_actions = 16, nt = None, dt =None, F =None, startpos = None, endpos = None, Test_grid =False):
    
    global state_list
    global base_path
    global save_path

    print("Building Sparse Model")
    t1 = time.time()
    #setup grid
    print("input to build_sparse_trans_model:\n")
    print("n_actions", n_actions)
    print("nt, dt", nt, dt)

    g, xs, ys, X, Y, vel_field_data, nmodes, num_rzns, path_mat, setup_params, setup_param_str = setup_grid(num_actions = n_actions, nt = nt, Test_grid= Test_grid)

    print("xs: ",xs)
    print("ys", ys)

    all_u_mat, all_v_mat, all_ui_mat, all_vi_mat, all_Yi = vel_field_data
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
    nrzns = num_rzns
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