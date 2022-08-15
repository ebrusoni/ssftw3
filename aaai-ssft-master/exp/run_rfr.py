import os
from unittest import result

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from sacred import Experiment
from tempfile import NamedTemporaryFile
import pandas as pd

import func_timeout as to

import sdsft
from sdsft import treesutils

from exp.ingredients import model
from exp.ingredients import tree_dataset as dataset

experiment = Experiment('training', ingredients=[model.ingredient, dataset.ingredient])

@experiment.config
def cfg():
    n_samples = 10000 #number of samples for computing the error estimates
    timeout = 144*3600 # 144 h timeout

@experiment.automain
def run(n_samples, timeout ,_run, _log):
    result = {}
    # Get data
    s, n = dataset.get_instance()
    bins = dataset.get_bins_no()
    test_set = dataset.get_test()

    ft = model.get_instance(n)
    
    try: 
        start = time.time()
        estimate = to.func_timeout(timeout, ft.transform, args=[s])
        end = time.time()
        print('estimated')
        gt_vec, est_vec = sdsft.eval_sf(s, estimate, n, n_samples = n_samples, err_type='raw')
        rel_rfr, mae_rfr, inf_rfr = treesutils.get_stats(gt_vec, est_vec)       

        n_queries = s.call_counter
        t = end-start
        print('estimate vs RFR      -> mae: %f, rel: %f, n_q: %d, coefs: %d, t: %f'%(mae_rfr, rel_rfr, n_queries, estimate.coefs.shape[0], t))

        test_bin = test_set[:, :-1]
        crit_temp_test = test_set[:, -1]
        s_test = lambda ind: (crit_temp_test[ treesutils.find_idx(test_bin, ind) ])
        s_test = sdsft.common.WrapSetFunction(s_test, use_loop=True)
        gt_vec, est_vec = sdsft.eval_sf(s_test, estimate, n, n_samples = n_samples, err_type='raw', custom_samples=test_bin)
        rel_test, mae_test, inf_test = treesutils.get_stats(gt_vec, est_vec)
        print('estimate vs test set -> mae: %f, rel: %f, n_q: %d, coefs: %d, t: %f'%(mae_test, rel_test, n_queries, estimate.coefs.shape[0], t))

        
        gt_vec, est_vec = sdsft.eval_sf(s_test, s, n, n_samples = n_samples, err_type='raw', custom_samples=test_bin)
        rel, mae, inf = treesutils.get_stats(gt_vec, est_vec)
        print('s vs test set        -> mae: %f, rel: %f'%(mae, rel))



        result['rel_rfr'] = result.get('rel_rfr', []) + [rel_rfr]
        result['mae_rfr'] = result.get('mae_rfr', []) + [mae_rfr]
        result['rel_test'] = result.get('rel_test', []) + [rel_test]
        result['mae_test'] = result.get('mae_test', []) + [mae_test]
        result['n_queries'] = result.get('n_queries', []) + [n_queries]
        result['time'] = result.get('time', []) + [t]
        result['freqs'] = result.get('freqs', []) + [estimate.freqs.tolist()]
        result['coefs'] = result.get('coefs', []) + [estimate.coefs.tolist()]
        #print('mae %f, rel %f, n_q %d, t %f'%(mae, rel, n_queries, t), end='\r')
        _run.log_scalar('k', len(estimate.coefs))

    except to.FunctionTimedOut:
        gt_vec, est_vec = 'timeout', 'timeout'
        t = 'timeout'
        rel = 'timeout'
        mae = 'timeout'
        n_queries = 'timeout'
        inf = 'timeout'

        result['rel'] = result.get('rel', []) + [rel]
        result['mae'] = result.get('mae', []) + [mae]
        result['n_queries'] = result.get('n_queries', []) + [n_queries]
        result['time'] = result.get('time', []) + [t]
        result['freqs'] = result.get('freqs', []) + ['timeout']
        result['coefs'] = result.get('coefs', []) + ['timeout']
        print('%d seconds timeout reached'%timeout, end='\r')
        
    
    
    _run.log_scalar('rel_rfr', rel_rfr)
    _run.log_scalar('mae_rfr', mae_rfr)
    _run.log_scalar('rel_test', rel_test)
    _run.log_scalar('mae_test', mae_test)
    _run.log_scalar('n_queries', n_queries)
    _run.log_scalar('time', t)
    _run.log_scalar('inf_rfr', inf_rfr)
    _run.log_scalar('inf_test', inf_test)
    
    #return result