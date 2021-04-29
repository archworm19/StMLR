
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
    # TODO: other data


    ## Tree Boosting Analysis

    # set indicator
    fn_set = 'zim_triple'
    Y2 = Y_zim
    inp2 = inp_zim # only needed for expansion

    # prediction type:
    fn_pred = 'RA'
    targ_cells = np.array([0,1])

    # add raw stimulus pattern
    for i in range(len(Y2)):
        Y2[i] = np.hstack((Y2[i], inp2[i][:,:1]))
        # TODO: other dataset???

    # general params: 
    in_cells = np.array([0,1,2,3])
    num_tree_cell = len(in_cells)
    in_cells_offset = np.array([6]) # NOTE: this was innapropriately set for other stuff!!!!

    if(fn_pred == 'DV'):
        basemus, X, worm_ids, t0s = build_tensors(Y2, targ_cells, in_cells, in_cells_offset, hist_len=24, dt=8)
        # get labels and filtered X
        olab = label_basemus(basemus, thrs=[-.02, .02])

    else:
        basemus, X, worm_ids, t0s = build_tensors(Y2, targ_cells, in_cells, in_cells_offset, hist_len=24, dt=6)
        # get labels and filtered X
        olab = label_basemus(basemus)

    Xf,fb = filterX(X)

    np.save('timing_fb.npy', fb)

    # hyper set handling:
    try:
        hyper_inds = np.load(fn_set + fn_pred + 'hyper_inds') > 0.5
    except:
        hyper_inds = npr.rand(np.shape(Xf)[0]) < 0.3
        np.save(fn_set + fn_pred + 'hyper_inds', hyper_inds*1)

    # try loading train/test sets:
    try:
        print('loading sets')
        train_sets = np.load(fn_set + fn_pred + '_trainsets.npy') > 0.5
        test_sets = np.load(fn_set + fn_pred + '_testsets.npy') > 0.5
    except:
        num_boot = 1000
        trainable_inds = np.ones((np.shape(Xf)[0])) > 0.5
        testable_inds = np.logical_not(hyper_inds)
        train_sets, test_sets = generate_traintest(np.shape(Xf)[0], num_boot, trainable_inds, testable_inds)
        np.save(fn_set + fn_pred + '_trainsets.npy', train_sets*1)
        np.save(fn_set + fn_pred + '_testsets.npy', test_sets*1)
        train_sets = train_sets > 0.5
        test_sets = test_sets > 0.5


    # Xf network
    Xf_net = Xf[:,:,:4,:]
    # Xf stim
    if(fn_pred == 'DV'):
        Xf_stim = Xf[:,:,6:,:]
    else:
        Xf_stim = Xf[:,:,4:6,:]

    """
    ## experiment: with stimulus context vs. without
    # without ~ ZIM, 2 boost found to be best
    mode = 2
    run_id = fn_set + fn_pred + 'mode' + str(mode)
    rc = get_run_config(mode, run_id)
    rc['tree_depth'] = [2,1]
    add_dat_rc(rc, hyper_inds, train_sets, test_sets, Xf_net, Xf_stim, worm_ids, olab)
   
    # mode 3 + 2state
    mode = 3
    run_id = fn_set + fn_pred + 'mode' + str(mode) + '_d1'
    rc2 = get_run_config(mode, run_id)
    rc2['tree_depth'] = [2,1]
    add_dat_rc(rc2, hyper_inds, train_sets, test_sets, Xf_net, Xf_stim, worm_ids, olab)

    # mode 3 + 4state
    mode = 3
    run_id = fn_set + fn_pred + 'mode' + str(mode) + '_d2'
    rc3 = get_run_config(mode, run_id)
    rc3['tree_depth'] = [2,2]
    add_dat_rc(rc3, hyper_inds, train_sets, test_sets, Xf_net, Xf_stim, worm_ids, olab)
    """

    ## experiment: fit to big set... refit to small set == buffer-buffer
    # ... TODO: migrate to using input
    #

    # dstruct contains all the different rcs:
    dstruct = [rc, rc2, rc3]

    # wrapper for boost cross-validation:
    def bcb(ds):
        return boot_cross_boosted(ds)

    with Pool(len(dstruct)) as p:
        p.map(bcb, dstruct)




