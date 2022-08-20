import os
from unittest import result

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from sacred import Experiment
from tempfile import NamedTemporaryFile
import pandas as pd
from sympy import fwht, ifwht

import func_timeout as to

import sdsft

from exp.ingredients import model
from exp.ingredients import preferencefun_dataset as dataset

experiment = Experiment('training', ingredients=[model.ingredient, dataset.ingredient])

@experiment.config
def cfg():
    n_samples = 10000 #number of samples for computing the error estimates
    timeout = 144*3600 # 144 h timeout

@experiment.automain
def run(n_samples, timeout, _run, _log):
    result = {}
    # Get data
    s, n = dataset.get_instance()
    # Get model
    ft = model.get_instance(n)
    try: 
        start = time.time()
        estimate = to.func_timeout(timeout, ft.transform, args=[s])
        end = time.time()
        print('estimated')
        gt_vec, est_vec = sdsft.eval_sf(s, estimate, n, n_samples = n_samples, err_type='raw')
        rel = np.linalg.norm(gt_vec - est_vec)/np.linalg.norm(gt_vec)
        mae = np.mean(np.abs(gt_vec - est_vec))
        inf = np.linalg.norm(gt_vec - est_vec, ord=np.inf)        

        n_queries = s.call_counter
        t = end-start
        print('mae: %f, rel: %f, n_q: %d, coefs: %d, t: %f'%(mae, rel, n_queries, estimate.coefs.shape[0], t))
        result['rel'] = result.get('rel', []) + [rel]
        result['mae'] = result.get('mae', []) + [mae]
        result['n_queries'] = result.get('n_queries', []) + [n_queries]
        result['time'] = result.get('time', []) + [t]
        result['freqs'] = result.get('freqs', []) + [estimate.freqs.tolist()]
        result['coefs'] = result.get('coefs', []) + [estimate.coefs.tolist()]
        print('mae %f, rel %f, n_q %d, t %f'%(mae, rel, n_queries, t), end='\r')
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
        
     
    _run.log_scalar('rel', rel)
    _run.log_scalar('mae', mae)
    _run.log_scalar('n_queries', n_queries)
    _run.log_scalar('time', t)
    _run.log_scalar('inf', inf)
    
    return result