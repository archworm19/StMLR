
if(__name__ == '__main__'):

    import numpy as np
    import numpy.random as npr
    import os
    import pylab as plt

    from master_timing import boot_cross_boosted
    from master_timing import pulse_expand
    from master_timing import build_tensors
    from master_timing import label_basemus
    from master_timing import filterX
    from master_timing import generate_traintest

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


    ## run params:
    tree_depths = [[4,0],[4,2],[4,2],[4,2],[4,2]]
    tree_widths = [[2,2],[2,2],[2,2],[2,2],[2,2]]
    l1_trees = [.01,.01,.01,.01,.01]
    l1_mlr_xf1s = [.05,.05,.05,.05,.05]
    l1_mlr_xf2s = [.1, .1, .1, .1, .1]
    l1_mlr_wids = [.1, .1, .1, .1, .1]
    lrs = [[], [], [10, 4], [20, 4], [20, 8]]


    ## Tree Boosting Analysis

    # set indicator
    fn_set = 'zim'
    Y2 = Y_zim
    inp2 = inp_zim # only needed for expansion

    # prediction type:
    fn_pred = 'RA'
    targ_cells = np.array([0,1])

    # expand cells ~ On cells + large on pulses, Off cells + large off pulses
    for i in range(len(Y2)):
        # ON cells
        eon = pulse_expand(Y2[i][:,4], inp2[i][:,0], 15, 100000)
        # OFF cells
        eoff = pulse_expand(Y2[i][:,5], 1-inp2[i][:,0], 15, 100000)
        Y2[i] = np.hstack((Y2[i],eon[:,None],eoff[:,None]))

    # general params: 
    in_cells = np.array([0,1,2,3])
    num_tree_cell = len(in_cells)
    in_cells_offset = np.array([4,5,6,7])

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


    # boost stim 4-state + low-rank
    futures = []
    for i in range(len(tree_depths)):
        # get string name
        fn_str = fn_set + fn_pred + 'depths' + str(tree_depths[i]) + 'LRs'
        + str(lrs[i])
        futures.append(boot_cross_boosted.remote(hyper_inds, train_sets,
            test_sets, Xf_net, Xf_stim, worm_ids, olab, tree_depths[i],
            tree_widths[i], l1_trees[i], l1_mlr_xf1=l1_mlr_xf1s[i],
            l1_mlr_xf2=l1_mlr_xf2s[i], l1_mlr_wid=l1_mlr_wids[i], num_model=25,
            num_epoch=30, mode=modes[i], fn_str=fn_str, lrs=lrs[i]))
    print(ray.get(futures))




