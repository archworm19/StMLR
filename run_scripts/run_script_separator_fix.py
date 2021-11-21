import numpy as np
import numpy.random as npr
import os
import pylab as plt
from multiprocessing import Pool
#from multiprocessing import shared_memory
import copy

from ..model_wrappers import master_separator as ms
from ..model_wrappers import core_utils 
from ..models import separator
from ..models import core_models

# enforces run on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# remake separator
# keep gating tree --> make new MLR object
# why? if sets have different numbers of worms --> wid_struct will be different shape
# --> different number of xdims in drive models
def remake_separator(S, rc):
    num_models = S.MLR.num_models
    num_state = S.MLR.num_state
    submodels_per_state = S.MLR.models_per_state
    output_cells = S.MLR.output_cells
    output_classes = S.MLR.output_classes   
    l1_mask = rc['l1_mlr_mask']
    l2_state_mask = rc['l2_state_mask']

    # xdim:
    xdim = np.shape(rc['Xf_drive'])[-1]

    print('xdim = ' + str(xdim))

    MLRC = separator.MultiLogRegComp(num_models, num_state, submodels_per_state, xdim, output_cells, output_classes, \
        l1_mask, l2_state_mask)    

    # make new separator:
    Snew = separator.Separator(S.F, MLRC, rc['comparison_filter'])
    return Snew



# TODO: run condition function:
# NOTE: this is super specific
def run_2cond(Y2, targ_cells, in_cells, in_cells_offset, fn_cond, fn_pred, S=[], num_boot=1000): 

    # replace nans with 0s in Y2:
    for Yc in Y2:
        naninds = np.isnan(Yc)
        Yc[naninds] = 0.0

    ## Build important tensors for each animal

    # dt = T2 (length of prediction window)
    # hist = T1 (length of input window)
    basemus, X, worm_ids, t0s = core_utils.build_tensors(Y2, targ_cells, in_cells, in_cells_offset, hist_len=24, dt=6)

    # get labels and filtered X
    olab = core_utils.label_basemus(basemus, thrs=[-.06, -.01, .01, .06])
    Xf,fb = core_utils.filterX(X)

    ## 2 Condition experiment
    # with and without stim
    Xf_drive_full = copy.deepcopy(Xf)
    Xf_drive_ko = copy.deepcopy(Xf)
    Xf_drive_ko[:,:,4:,:] = 0.0

    # make Xf gate and Xf drive:
    # pre xdim... still separates cells and filters
    Xf_gate = copy.deepcopy(Xf)
    Xf_drive = np.concatenate((Xf_drive_full[:,:,None,:,:], Xf_drive_ko[:,:,None,:,:]), axis=2)

    # convert to xdim (flatten across cells and filters)
    [num_block, Tper, num_cell, subx] = np.shape(Xf_gate)
    Xf_gate = np.reshape(Xf_gate, (num_block, Tper, num_cell*subx))
    Xf_drive = np.reshape(Xf_drive, (num_block, Tper, 2, num_cell*subx)) # 2 refers to 2 conditions

    # combine Xf_drive with worm_ids
    # both submodels should get access:
    worm_ids2 = worm_ids[:,:,None,:]
    worm_ids2 = np.tile(worm_ids2, (1,1,2,1)) # 2 refers to 2 conditions
    Xf_drive = np.concatenate((Xf_drive, worm_ids2), axis=3)

    ## make l1 masks for Xf_drive = l1_mlr_mask
    # l1 mask for different cells:
    l1_base_scale = .01
    l1_stim_scale = .15
    l1_cells = np.ones((num_cell, subx))
    l1_cells[:4,:] = l1_base_scale
    l1_cells[4:,:] = l1_stim_scale
    # wid mask?
    l1_wid_scale = .05
    l1_wid = np.ones((np.shape(worm_ids)[-1]))
    l1_wid[:] = l1_wid_scale

    # combine into single mask --> l1_mlr_mask
    l1_mlr_mask = np.hstack((np.reshape(l1_cells,-1), l1_wid))

    ## cross experiment
    # ~50% goes into hyperparam set
    hyper_inds = npr.random(np.shape(Xf_gate)[0]) < 0.45
    # cross-validation train/test set generation:
    # everything trainable
    trainable_inds = np.ones((np.shape(Xf_gate)[0])) > 0.5
    # testable = not hyper set
    testable_inds = np.logical_not(hyper_inds)
    train_sets, test_sets = core_utils.generate_traintest(np.shape(Xf_gate)[0], num_boot, trainable_inds, testable_inds, test_perc=0.5)
    train_sets = train_sets > 0.5
    test_sets = test_sets > 0.5

    print('training vs. test')
    print(np.sum(train_sets))
    print(np.sum(test_sets))
    print(np.sum(train_sets == test_sets))

    # get run configs:
    rc = ms.get_run_config('sep_' + fn_cond + fn_pred + str(l1_stim_scale))
    
    # fill out run config
    rc['hyper_inds'] = hyper_inds
    rc['train_sets'] = train_sets
    rc['test_sets'] = test_sets
    rc['Xf_gate'] = Xf_gate
    rc['Xf_drive'] = Xf_drive
    rc['olabs'] = olab
    rc['l1_mlr_mask'] = l1_mlr_mask

    # basic comparison filter ~ fit for 0th submodel
    rc['comparison_filter'] = np.array([1,0]).astype(np.float32)

    # if using an existing separator
    # --> remake the drive to account for different set sizes:
    if(isinstance(S, separator.Separator)):
        S = remake_separator(S, rc)

    # run out-of-bootstrap cross-validation on set
    # makes a directory and saves important info there
    # boot_cross optionally takes in a pre-trained separator class
    S = ms.boot_cross(rc, S=S)
    return S



if(__name__ == '__main__'):
    
    ## Multi Separator run
    # simple version:
    # fit whole model base dataset --> fit MLR submodels only for each other dataset

    # Fix: loads from old, saved separator array
    # this should be temporary now that core issue has been patched!

    # load old forest:
    gate_dir = '/snl/scratch/ztcecere/TEMP_SEPARATE3/sep_zimRA0.15'
    gate_vars = np.load(os.path.join(gate_dir, 'gate_vars.npy'))
    # all boots are identical --> take 0th
    gate_vars = gate_vars[0] # --> num_branch x num_tree x tree_width x xdim
    # use all trees:
    use_trees = np.ones((np.shape(gate_vars)[1])) > 0.5
    # initialize forest from old forest:
    F = core_models.Forest(2, 2, len(use_trees), np.shape(gate_vars)[-1], 0.1, 0.0, tree_struct_ar=gate_vars, use_trees=use_trees)
    # create new MLR object 
    # NOTE: just has to build... will be remade later anyway
    MLRC = separator.MultiLogRegComp(len(use_trees), 4, 2, 1, 2, 5, \
        0.0, 0.0)    
    S = separator.Separator(F, MLRC, np.array([1,0]))

    ## load dat: 
    import sys
    #sys.path.append('/home/ztcecere/CodeRepository/PD/')
    sys.path.append('/snl/scratch/ztcecere/PD')
    import data_loader
    #rdir = '/data/ProcAiryData'
    rdir = '/home/ztcecere/ProcAiryData'
    inp, Y, inp_zim, Y_zim = data_loader.load_data(rdir)
    # load ztc buffer-to-buffer
    inp_ztcbuf, Y_ztcbuf = data_loader.load_data_ztcNoStim(rdir)
    # load jh buffer-to-buffer
    inp_jhbuf, Y_jhbuf = data_loader.load_data_jh_buf2buf_trial2(rdir) 

    # order matters
    # TODO: only need to run the bufs for fix
    Y_l = [Y_ztcbuf, Y_jhbuf]
    fn_conds = ['zimbuf', 'javbuf']

    # Forecasting specification
    # target_cells = which cells to predict future activity of:
    targ_cells = np.array([0,1])
    fn_pred = 'RA'

    ## in_cells ~ which cells should be used as inputs to forecasting model
    # timing setup: t_break = time breakpoint
    # prediction window: t_break...t_break+T2
    # pre-window: t_break-T1...t_break
    # offset window: t_break-T1+T2...t_break+T2
    #
    # cells in pre-window
    in_cells = np.array([0,1,2,3])
    # cells in offset window:
    in_cells_offset = np.array([4,5])

    # number of bootstraps for out-of-bootstrap cross-validation 
    num_boot = 1000

    for i in range(len(Y_l)):
        S = run_2cond(Y_l[i], targ_cells, in_cells, in_cells_offset, fn_conds[i], fn_pred, S=S, num_boot=1000)


