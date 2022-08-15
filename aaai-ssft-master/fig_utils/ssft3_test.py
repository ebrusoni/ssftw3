from cmath import e
from pickle import TRUE
import numpy as np
import sdsft.transforms.canonical as can
from sdsft.common import WrapSetFunction
import sdsft.common as com
import exp.fitness_utils as futils
from exp.ingredients import fitness_dataset
import scipy.io
import sys
import tqdm
import itertools
import ck.kernel as ck

sys.path.append('.')

import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def random_s(n, k, model, p=0.5):
    freqs = []
    coeffs = []
    for i in range(k):
        # freq = np.random.randint(2,size=n)
        freq =  np.random.binomial(1, p, size=(n,))
        freqs.append(freq)
        coef = np.random.normal(0, 10)
        coeffs.append(coef)
    print(f'objective freqs: {freqs} \nobjective coeffs: {coeffs}')
    if model == 'W3':
        return com.SparseDSFT3Function(np.asarray(freqs), np.asarray(coeffs), model='W3')
    if model == '3' : 
        return com.SparseDSFT3Function(np.asarray(freqs), np.asarray(coeffs), model='3')
    if model == '4':
        return com.SparseDSFT4Function(np.asarray(freqs), np.asarray(coeffs))
    #return com.SparseDSFT3Function(np.array(freqs), np.array(coeffs), model='3')

def s(arr):
    arr = np.array(arr, dtype=bool)
    s = [('FFF', [0]), ('TFF', [2]), ('FTF', [2]), ('TTF', [2]), ('FFT', [0]), ('TFT', [4]), ('FTT', [3]), ('TTT',[4])]
    ds = dict(s)
    key=''
    for i in range(arr.shape[0]):
        if arr[i]:
            key = key + 'T'
        else:
            key = key + 'F'
    return np.array(ds[key])[0]

def pref_func(idx, network):
    if np.any(idx):
        maxs = np.max(network[idx.astype(np.bool)], axis=0)
        assert(maxs.shape == (3424,))
        return np.sum(maxs)
    else:
        return 0.0

def test_sparsity(n, k):
    rounds = 10
    for p in np.arange(0, 1, 0.1):
        avg_coefs = 0
        print(f'p: {p}')
        i=0
        #with tqdm.tqdm(total=rounds, file=sys.stdout) as pbar:
        for i in range(rounds):
            rev_s = random_s(n, k, p)
            sample_s = lambda idx: np.sqrt(3)*rev_s(idx)
            sample_s = com.WrapSetFunction(sample_s)
            SSFT3 = can.SparseSFT(n, flag_general=False, flag_print=False, k_max=None, model = 'W3', eps=1e-6)
            estimate2 = SSFT3.transform(sample_s)
            avg_coefs += estimate2.coefs.shape[0]
            err2 = com.eval_sf(sample_s, estimate2, n, n_samples=100000, err_type='rel')
            print(f'roundrel error {err2}, coefs: {estimate2.coefs.shape[0]}')
        print(f'avg coefs for p={p}: {avg_coefs/rounds}')

def compile_run(flags):
    r=ck.access({'action':'compile', 'module_uoa':'program', 'data_uoa':'cbench-automotive-susan', 
                'flags': flags})
    if r['return']>0: return r # unified error handling 

    # Equivalent of "ck run program:image-corner-detection --env.OMP_NUM_THREADS=1
    r=ck.access({'action':'run', 'module_uoa':'program', 'data_uoa':'cbench-automotive-susan', 
                 'env':{'OMP_NUM_THREADS':1}, 'dataset_uoa':'b2130844c38e4a56', 'cmd_key':'corners',
                 'console':'no'})
    if r['return']>0: return r # unified error handling 

    print(r['characteristics']['execution_time'])
    return r['characteristics']['execution_time']

def load_susan():
    gcc_flags = np.array(['-O2', '-fno-tree-loop-vectorize' , '-fno-unsafe-math-optimizations',  
                '-fprefetch-loop-arrays',  '-fsigned-zeros',  '-funroll-all-loops',  
                '-ffast-math'])
    n = len(gcc_flags)
    idx = np.array([1,1,0,0,1,0,1])
    # print(gcc_flags[idx.astype(bool)])
    # print(' '.join(gcc_flags[idx.astype(bool)]))
    set_function = WrapSetFunction(lambda idx: compile_run('-O3 -Q --help=optimizers'), use_call_dict=True)

    return set_function, n

def load_other():
    pass


# wrapped_s = com.WrapSetFunction(s, use_loop=True)
# SSFT3 = can.SparseSFT(3, flag_general=False, flag_print=True, model='W3')
# estimate = SSFT3.transform(wrapped_s)
# print(estimate.freqs)
# print(estimate.coefs)
# err = com.eval_sf(wrapped_s, estimate, 3, n_samples=1000, err_type='rel')
# print(f'error: {err}')

# mat = scipy.io.loadmat('./exp/datasets/preference_function/water_imp1000.mat')
# arr = mat['Z1'].toarray()
# maxs = np.max(arr, axis=1)
# idxs = np.argsort(-maxs)

# k = 20
# network = arr[idxs[:k]]
# wrapped_s = com.WrapSetFunction(lambda idx : pref_func(idx, network), use_loop=True)
# SSFT3 = can.SparseSFT(k, flag_general=False, flag_print=True, model='3', eps=1e-3, k_max=1000)
# estimate = SSFT3.transform(wrapped_s)
# err = com.eval_sf(wrapped_s, estimate, k, n_samples=1000, err_type='rel')
# print(f'relative error: {err}')
# print(f'no. of coefs: {estimate.coefs.shape}')

# n=10
# k=10
# sample_s = random_s(n, k, p=0.5)
# sample_s = WrapSetFunction(s, use_loop=True)
# #print(list(WrapSetFunction(sample_s).to_vector(n)))
# n = 3
# SSFT3 = can.SparseSFT(n, flag_general=True, flag_print=True, k_max=None, model = '3', eps=1e-12)
# estimate2 = SSFT3.transform(sample_s)
# print(f'coefs: {estimate2.coefs}')
# print(f'coefs: {estimate2.coefs.shape[0]}')
# print(f'freqs:\n {estimate2.freqs}')
# err2 = com.eval_sf(sample_s, estimate2, n, n_samples=100000, err_type='rel')
# print(f'error: {err2}')
# print(estimate2)
# idxs = np.array(list(itertools.product([0,1], repeat=3)))
# print(idxs)
# print(estimate2(idxs))

#test_sparsity(10, 50)

# sf, n = fitness_dataset.load_khan()
# SSFT = can.SparseSFT(n, flag_general=False, flag_print=False, k_max=None, model='3')
# some_not_full = False
# for perm in list(itertools.permutations(range(n))):
#     print(perm)
#     sf_perm = lambda idx : sf.s(idx[list(perm)])
#     sf_perm_wrap = com.WrapSetFunction(sf_perm, use_loop=True)
#     estimate = SSFT.transform(sf_perm_wrap)

#     #print(f'coefs: \n{estimate.coefs}')
#     err = com.eval_sf(sf_perm_wrap, estimate, n, n_samples=100, err_type='rel')
#     # if(err < 1e-4):
#     #     print(estimate.coefs)
#     if(len(estimate.coefs) < 2**n):
#         print(f'coefs: {len(estimate.coefs)}')
#         some_not_full = True
#     else:
#         print('ALL COEFS')
#     print(f'error: {err}')
# print(some_not_full)
    
# set_function, n=  load_susan()
# SSFT = can.SparseSFT(n, flag_general=False, flag_print=True, k_max=10, model='W3')
# estimate = SSFT.transform(set_function)
# err = com.eval_sf(set_function, estimate, n, n_samples=10, err_type='rel')
# print(f'relative error: {err}')


