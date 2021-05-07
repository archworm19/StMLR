
if(__name__ == '__main__'):

    import numpy as np
    import numpy.random as npr
    import os
    import pylab as plt
    from multiprocessing import Pool
    import copy

    from master_timing import boot_cross_boosted
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


    ## load dat: TODO
    import sys
    #sys.path.append('/home/ztcecere/CodeRepository/PD/')
    sys.path.append('/snl/scratch/ztcecere/PD')
    import data_loader
    #rdir = '/data/ProcAiryData'
    rdir = '/home/ztcecere/ProcAiryData'
    inp, Y, inp_zim, Y_zim = data_loader.load_data(rdir)


    ## Tree Boosting Analysis

    # set indicator
    fn_set = 'all'
    Y2 = [Y, Y_zim]
    inp2 = [inp, inp_zim]

    # prediction type:
    fn_pred = 'RA'
    targ_cells = np.array([0,1])

    # add On and Off stimuli:
    for i, Yc in enumerate(Y2):
        for j, Ysub in enumerate(Yc):
            inpc = inp2[i][j][:,0]
            off_inpc = 1 - inpc
            first_on = np.where((inpc == 1))[0][0]
            off_inpc[:first_on] = 0
            Y2[i][j] = np.hstack((Ysub, inpc[:,None], off_inpc[:,None]))

    # general params: 
    in_cells = np.array([0,1,2,3])
    num_tree_cell = len(in_cells)
    in_cells_offset = np.array([4,5,6,7])

    # build tensors for each subset:
    Xfs_l, olabs_l, worm_ids_l, fb = [], [], [], []
    # iter thru conditions:
    for i, Yc in enumerate(Y2): 
        basemus, X, worm_ids, t0s = build_tensors(Yc, targ_cells, in_cells, in_cells_offset, hist_len=24, dt=6)

        # get labels and filtered X
        olab = label_basemus(basemus, thrs=[-.06, -.02, .02, .06])
        Xf,fb = filterX(X)

        # save:
        Xfs_l.append(Xf)
        olabs_l.append(olab)
        worm_ids_l.append(worm_ids)

    np.save('timing_fb.npy', fb)

    # hyper set handling:
    try:
        hyper_inds = np.load(fn_set + fn_pred + 'hyper_inds') > 0.5
    except:
        tot_size = sum([np.shape(xfi)[0] for xfi in Xfs_l])
        hyper_inds = npr.rand(tot_size) < 0.3
        np.save(fn_set + fn_pred + 'hyper_inds', hyper_inds*1)

    # try loading train/test sets:
    try:
        print('loading sets')
        train_sets = np.load(fn_set + fn_pred + '_trainsets.npy') > 0.5
        test_sets = np.load(fn_set + fn_pred + '_testsets.npy') > 0.5
    except:
        num_boot = 1000
        tot_size = sum([np.shape(xfi)[0] for xfi in Xfs_l])
        trainable_inds = np.ones(tot_size) > 0.5
        testable_inds = np.logical_not(hyper_inds)
        train_sets, test_sets = generate_traintest(tot_size, num_boot, trainable_inds, testable_inds)
        np.save(fn_set + fn_pred + '_trainsets.npy', train_sets*1)
        np.save(fn_set + fn_pred + '_testsets.npy', test_sets*1)
        train_sets = train_sets > 0.5
        test_sets = test_sets > 0.5


    # Xf network
    Xf_net = [Xf[:,:,:4,:] for Xf in Xfs_l]
    # Xf stim ~ raw 
    Xf_stim = [Xf[:,:,6:,:] for Xf in Xfs_l]

    # stack olabs --> olab:
    olab = np.vstack(olabs_l)

    ## experiment: with stimulus context vs. without
 
    # depth 1
    mode = 4 # ~ random slope
    run_id = fn_set + fn_pred + 'mode' + str(mode) + '_d1'
    rc = get_run_config(mode, run_id)
    rc['tree_depth'] = [2,1]
    add_dat_rc(rc, hyper_inds, train_sets, test_sets, Xf_net, Xf_stim, worm_ids_l, olab)
    rc['l1_tree'] = [.01, 2.0]


    # dstruct contains all the different rcs:
    dstruct = [rc]


    # wrapper for boost cross-validation:
    def bcb(ds):
        return boot_cross_boosted(ds)

    with Pool(len(dstruct)) as p:
        p.map(bcb, dstruct)



