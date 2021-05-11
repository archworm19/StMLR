
if(__name__ == '__main__'):

    # 3 = triple fit

    import numpy as np
    import numpy.random as npr
    import os
    import pylab as plt
    from multiprocessing import Pool
    import copy

    from master_timing import boot_cross_boosted
    from master_timing import triple_fit 
    from master_timing import pulse_expand
    from master_timing import build_tensors
    from master_timing import label_basemus
    from master_timing import filterX
    from master_timing import generate_traintest
    from master_timing import get_run_config
    from master_timing import add_dat_rc

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    plt.rcParams.update({'font.size': 14})

    VOLS_PER_SECOND = 1.5
    CELL_MANIFEST = ['AVA', 'RME', 'SMDV', 'SMDD', 'ON', 'OFF']


    ## load dat: 
    import sys
    sys.path.append('/snl/scratch/ztcecere/PD')
    import data_loader
    rdir = '/home/ztcecere/ProcAiryData'
    inp, Y, inp_zim, Y_zim = data_loader.load_data(rdir)
    # NoStim data
    inp_nos, Y_nos = data_loader.load_data_ztcNoStim(rdir)


    ## Tree Boosting Analysis

    # set indicator
    fn_set = 'zim_triple'
    Y2 = Y_zim
    inp2 = inp_zim # only needed for expansion
    Y3 = Y_nos
    inp3 = inp_nos

    # prediction type:
    fn_pred = 'RA'
    targ_cells = np.array([0,1])

    # add raw stimulus pattern
    # and OFF
    for i in range(len(Y2)):
        # get off pattern:
        first_on = np.where(inp2[i][:,0]==1)[0][0]
        ioff = 1 - inp2[i][:,:1]
        ioff[:first_on,:] = 0
        Y2[i] = np.hstack((Y2[i], inp2[i][:,:1], ioff))
        
    for i in range(len(Y3)):
        # get off pattern:
        ion = np.reshape(inp3[i],(-1,1))
        first_on = np.where(ion[:,0]==1)[0][0]
        ioff = 1 - ion
        ioff[:first_on,:] = 0
        Y3[i] = np.hstack((Y3[i], ion, ioff))

    # general params: 
    in_cells = np.array([0,1,2,3])
    num_tree_cell = len(in_cells)
    in_cells_offset = np.array([4,5,6,7]) 

    if(fn_pred == 'DV'):
        basemus, X, worm_ids, t0s = build_tensors(Y2, targ_cells, in_cells, in_cells_offset, hist_len=24, dt=8)
        basemus3, X3, worm_ids3, t0s3 = build_tensors(Y3, targ_cells, in_cells, in_cells_offset, hist_len=24, dt=8)
        # get labels and filtered X
        olab = label_basemus(basemus, thrs=[-.02, .02])
        olab3 = label_basemus(basemus3, thrs=[-.02, .02])

    else:
        basemus, X, worm_ids, t0s = build_tensors(Y2, targ_cells, in_cells, in_cells_offset, hist_len=24, dt=6)
        basemus3, X3, worm_ids3, t0s3 = build_tensors(Y3, targ_cells, in_cells, in_cells_offset, hist_len=24, dt=6)
        # get labels and filtered X
        olab = label_basemus(basemus)
        olab3 = label_basemus(basemus3)

    Xf,fb = filterX(X)
    Xf3,null = filterX(X3)

    np.save('timing_fb.npy', fb)

    # hyper set handling:
    try:
        hyper_inds = np.load(fn_set + fn_pred + 'hyper_inds') > 0.5
    except:
        hyper_inds = npr.rand(np.shape(Xf)[0]) < 0.3
        np.save(fn_set + fn_pred + 'hyper_inds', hyper_inds*1)

    # number of bootstraps
    num_boot = 1000

    # try loading train/test sets:
    try:
        print('loading sets')
        train_sets = np.load(fn_set + fn_pred + '_trainsets.npy') > 0.5
        test_sets = np.load(fn_set + fn_pred + '_testsets.npy') > 0.5
    except:
        trainable_inds = np.ones((np.shape(Xf)[0])) > 0.5
        testable_inds = np.logical_not(hyper_inds)
        train_sets, test_sets = generate_traintest(np.shape(Xf)[0], num_boot, trainable_inds, testable_inds)
        np.save(fn_set + fn_pred + '_trainsets.npy', train_sets*1)
        np.save(fn_set + fn_pred + '_testsets.npy', test_sets*1)
        train_sets = train_sets > 0.5
        test_sets = test_sets > 0.5


    try: 
        train_sets3 = np.load(fn_set + fn_pred + '_trainsets3.npy') > 0.5
        test_sets3 = np.load(fn_set + fn_pred + '_testsets3.npy') > 0.5
    except: 
        # repeat for 3 set:
        trainable_inds3 = np.ones((np.shape(Xf3)[0])) > 0.5
        testable_inds3 = np.ones((np.shape(Xf3)[0])) > 0.5
        train_sets3, test_sets3 = generate_traintest(np.shape(Xf3)[0], num_boot, trainable_inds3, testable_inds3)
        np.save(fn_set + fn_pred + '_trainsets3.npy', train_sets3*1)
        np.save(fn_set + fn_pred + '_testsets3.npy', test_sets3*1)
        train_sets3 = train_sets3 > 0.5
        test_sets3 = test_sets3 > 0.5


    # Xf network
    Xf_net = Xf[:,:,:4,:]
    # Xf stim ~ raw:
    Xf_stim = Xf[:,:,6:,:]
    # refit stuff:
    Xf_net_refit = Xf3[:,:,:4,:]
    Xf_stim_refit = Xf3[:,:,6:,:]

    # triple fit ~ ZIM RA   
    # only need 1 trial/run config
    mode = 2
    run_id = fn_set + fn_pred + 'mode' + str(mode) + '_3fit_ONOFF'
    rc = get_run_config(mode, run_id)
    rc['tree_depth'] = [2,1]
    add_dat_rc(rc, hyper_inds, train_sets, test_sets, Xf_net, Xf_stim, worm_ids, olab)
    rc['Xf_net_refit'] = Xf_net_refit
    rc['Xf_stim_refit'] = Xf_stim_refit
    rc['worm_ids_refit'] = worm_ids3
    rc['train_sets_refit'] = train_sets3
    rc['test_sets_refit'] = test_sets3
    rc['olab_refit'] = olab3

    ## Handle Worm Ids
    # worm_ids must be the same shape across worms:
    # get new shapes to add
    sh_add0 = [np.shape(rc['worm_ids'])[0], np.shape(rc['worm_ids'])[1], np.shape(rc['worm_ids_refit'])[2]-1]
    sh_add1 = [np.shape(rc['worm_ids_refit'])[0], np.shape(rc['worm_ids_refit'])[1], np.shape(rc['worm_ids'])[2]-1]
    # first set
    rc['worm_ids'] = np.concatenate((rc['worm_ids'], np.zeros((sh_add0))), axis=2)
    # second set
    rc['worm_ids_refit'] = np.concatenate((np.zeros((sh_add1)), rc['worm_ids_refit']), axis=2)

    # dstruct contains all the different rcs:
    dstruct = [rc]

    # wrapper for boost cross-validation:
    def bcb3(ds):
        return triple_fit(ds)

    with Pool(len(dstruct)) as p:
        p.map(bcb3, dstruct)




