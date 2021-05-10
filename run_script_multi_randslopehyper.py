import numpy as np
import numpy.random as npr
import os
import pylab as plt
from multiprocessing import Pool
from multiprocessing import shared_memory
import copy

from master_timing import boot_cross_boosted
from master_timing import boot_cross_hyper
from master_timing import pulse_expand
from master_timing import build_tensors
from master_timing import label_basemus
from master_timing import filterX
from master_timing import generate_traintest
from master_timing import get_run_config
from master_timing import add_dat_rc
from master_timing import new_run_config_axis


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

plt.rcParams.update({'font.size': 14})


#### Shared Memory Solution
# requires python3.8

# shared memory creation:
# returns: [name, shape, dtype]
def create_shm_obj(np_obj):
    # create new shm
    shm = shared_memory.SharedMemory(create=True, size=np_obj.nbytes)
    # create numpy array backed by shared memory:
    npb = np.ndarray(np_obj.shape, dtype=np_obj.dtype, buffer=shm.buf)
    # copy og data into shared memory:
    npb[:] = np_obj[:]
    return [shm.name, np.shape(npb), npb.dtype]


# convert numpy arrays to shared memory arrays
# writes shared memory info into run_config
# each data entry = [shared memory name, shape, dtype]
# NOTE: writes sharedmem info over nump array in run_config
def create_shm(rc):
    # numpy targets
    target_keys = ['hyper_inds', 'train_sets', 'test_sets', 'train_sets_hyper', 'test_sets_hyper'] 
    for tk in target_keys:
        if(tk in rc):
            rc[tk] = create_shm_obj(rc[tk])
    # list of numpy targets:
    target_keys = ['Xf_net', 'Xf_stim', 'worm_ids', 'olab'] 
    for tk in target_keys:
        if(tk in rc):
            rc_new = []
            for ar in rc[tk]:
                rc_new.append(create_shm_obj(ar))
            rc[tk] = rc_new
   

# convert shared memory name to numpy array
# 1. attached to shared memory object (via name)
# 2. attach numpy array buffer to shared memory object
# 3. copy reference for numpy array into run_config dictionary (replaces name)
# NOTE stored in run_config = [name, shape, dtype]
# NOTE: this should be called after mapping 
def convert_shmname_to_np(rc):
    # numpy targets:
    target_keys = ['hyper_inds', 'train_sets', 'test_sets', 'train_sets_hyper', 'test_sets_hyper'] 
    for tk in target_keys:
        if(tk in rc):
            # get info
            [name, shape, dtype] = rc[tk]
            # attach to shared memory:
            cur_shm = shared_memory.SharedMemory(name=name)
            # attach numpy array buffer to shared memory:
            ar = np.ndarray(shape, dtype=dtype, buffer=cur_shm.buf)
            # replace in run_config:
            rc[tk] = ar
    # targets: list of numpy arrays:
    target_keys = ['Xf_net', 'Xf_stim', 'worm_ids', 'olab'] 
    for tk in target_keys:
        if(tk in rc):
            cur_v = []
            for l in rc[tk]:
                # get info
                [name, shape, dtype] = l
                # attach to shared memory:
                cur_shm = shared_memory.SharedMemory(name=name)
                # attach numpy array buffer to shared memory:
                ar = np.ndarray(shape, dtype=dtype, buffer=cur_shm.buf)
                # add to current list:
                cur_v.append(ar)
            # replace in run_config:
            rc[tk] = cur_v


#### FileSystem Soln
# rc holds filenames --> each process loads and replaces
# NOTE: save complete path


def convert_np_to_fn(rc):
    target_keys = ['hyper_inds', 'train_sets', 'test_sets', 'train_sets_hyper', 'test_sets_hyper']
    root_dir = rc['dir_str']
    for tk in target_keys:
        if(tk in rc):
            full_path = os.path.join(root_dir, tk)
            np.save(full_path, rc[tk])
            rc[tk] = full_path

    target_keys = ['Xf_net', 'Xf_stim', 'worm_ids', 'olab'] 
    for tk in target_keys:
        if(tk in rc):
            full_path = os.path.join(root_dir, tk)
            np.savez(full_path, rc[tk])
            rc[tk] = full_path

# convert npz to list
# takes in the npz dictionary
def conv_npz_l(d):
    l = []
    for i in range(len(d.keys())):
        l.append(d['arr_' + str(i)])
    return l


# each process should call this to load the required data
def convert_fn_to_np(rc):
    target_keys = ['hyper_inds', 'train_sets', 'test_sets', 'train_sets_hyper', 'test_sets_hyper']
    for tk in target_keys:
        if(tk in rc):
            rc[tk] = np.load(rc[tk])

    target_keys = ['Xf_net', 'Xf_stim', 'worm_ids', 'olab'] 
    for tk in target_keys:
        if(tk in rc):
            rc[tk] = conv_npz_l(np.load(rc[tk]))


if(__name__ == '__main__'):


    VOLS_PER_SECOND = 1.5
    CELL_MANIFEST = ['AVA', 'RME', 'SMDV', 'SMDD', 'ON', 'OFF']


    ## load dat: 
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
    num_boot = 5

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
        train_sets_hyper = np.load(fn_set + fn_pred + 'train_sets_hyper') > 0.5
        test_sets_hyper = np.load(fn_set + fn_pred + 'test_sets_hyper') > 0.5
    except:
        tot_size = sum([np.shape(xfi)[0] for xfi in Xfs_l])
        hyper_inds = npr.rand(tot_size) < 0.3
        np.save(fn_set + fn_pred + 'hyper_inds', hyper_inds*1)
        # hyperparameter train/test set generation:
        train_sets_hyper, test_sets_hyper = generate_traintest(tot_size, 10, hyper_inds, hyper_inds, train_perc=0.95)
        np.save(fn_set + fn_pred + 'train_sets_hyper.npy', train_sets_hyper*1)
        np.save(fn_set + fn_pred + 'test_sets_hyper.npy', test_sets_hyper*1)
        train_sets_hyper = train_sets_hyper > 0.5
        test_sets_hyper = test_sets_hyper > 0.5
       

    # try loading train/test sets:
    try:
        print('loading sets')
        train_sets = np.load(fn_set + fn_pred + '_trainsets.npy') > 0.5
        test_sets = np.load(fn_set + fn_pred + '_testsets.npy') > 0.5
    except:
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

    # load numpy data structures into shared memory arrays
    sh_hyper_inds = shared_memory.SharedMemory(create=True, size=hyper_inds.nbytes)

    ## experiment: with stimulus context vs. without
 
    # base run_config:
    mode = 4 # ~ random slope
    run_id = fn_set + fn_pred + 'mode' + str(mode) + '_d1'
    rc = get_run_config(mode, run_id)
    rc['tree_depth'] = [2,1]
    add_dat_rc(rc, hyper_inds, train_sets, test_sets, Xf_net, Xf_stim, worm_ids_l, olabs_l, train_sets_hyper,
            test_sets_hyper) 
    rc['l1_tree'] = [.01, 2.0]


    # save big datastructures to lists... pass around filenames:
    convert_np_to_fn(rc)

    # dstruct contains all the different rcs:
    dstruct = [rc]

    # TODO/TESTING: get combos:
    s = 'l1_mlr_xf1' 
    vals = [[.05, 1.0], [.02, 1.0], [.02, 2.0], [.05, 2.0]]
    dstruct = new_run_config_axis(dstruct, s, vals)

    # hyperparameter testing
    def bch(rc): 
        convert_fn_to_np(rc)
        return boot_cross_hyper(rc)

    # TODO: there should be a way to limit the number of processes in the pool
    MAXPROC = 5
    with Pool(MAXPROC) as p:
        p.map(bch, dstruct)


