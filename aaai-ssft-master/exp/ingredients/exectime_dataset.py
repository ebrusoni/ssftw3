import inspect
import pandas as pd

from sacred import Ingredient

import functools
import sdsft
import numpy as np
from sdsft.common import WrapSetFunction
import ck.kernel as ck

ingredient = Ingredient('dataset')

@ingredient.config
def cfg():
    """Dataset configuration"""
    name = ''
    set_function = None
    n = None

def compile_run_susan(flags):
    #reference for ck python API: https://ck.readthedocs.io/en/latest/src/first-steps.html#use-ck-python-api
    r=ck.access({'action':'compile', 'module_uoa':'program', 'data_uoa':'cbench-automotive-susan', 
                'flags': flags, 'speed':'yes'})#  
    if r['return']>0: return r # unified error handling 

    # Equivalent of "ck run program:image-corner-detection --env.OMP_NUM_THREADS=1
    r=ck.access({'action':'run', 'module_uoa':'program', 'data_uoa':'cbench-automotive-susan', 
                 'env':{'OMP_NUM_THREADS':1}, 'dataset_uoa':'b2130844c38e4a56', 'cmd_key':'corners'})
    if r['return']>0: return r # unified error handling 

    print(r['characteristics']['execution_time'])
    return np.log(r['characteristics']['execution_time'])

def compile_run_jpeg(flags):
    #reference for ck python API: https://ck.readthedocs.io/en/latest/src/first-steps.html#use-ck-python-api
    r=ck.access({'action':'compile', 'module_uoa':'program', 'data_uoa':'cbench-consumer-jpeg-d', 
                'console':'no','flags': flags, 'speed':'yes'})#  cbench-automotive-susan
    if r['return']>0: return r # unified error handling 

    # Equivalent of "ck run program:image-corner-detection --env.OMP_NUM_THREADS=1
    r=ck.access({'action':'run', 'module_uoa':'program', 'data_uoa':'cbench-consumer-jpeg-d', 
                 'env':{'OMP_NUM_THREADS':1}, 'console':'no', 'dataset_uoa':'1aaaa23c44e588f9' })
    if r['return']>0: return r # unified error handling 

    print(r['characteristics']['execution_time'])
    return np.log(r['characteristics']['execution_time'])

def load_prog(prog ='susan', n=10):
    np.random.seed(0)
    O3_flags = np.loadtxt('/home/enri/code/ba/aaai-ssft-master/exp/datasets/gcc/O3fno.txt', dtype=str)#, delimiter='\n'
    print(len(O3_flags))
    # removed -fipa-modref, -fipa-reference-addressable, -fmove-loop-stores, -ffinite-loops, -fversion-loops-for-strides, changed -fvect-cost-model=very-cheap to -fvect-cost-model=cheap
    # because gcc9 doesn't recognize them
    #n = 2
    flags_idx = np.arange(n)
    #flags_idx = np.sort(np.random.choice(len(O3_flags), n, replace=False))
    print(flags_idx)
    gcc_flags = O3_flags[flags_idx]
    print(gcc_flags)
    # gcc_flags = O3_flags[:n]
    # rest = ' '.join(O3_flags[n:])
    #rest += ' -Q --help=optimizers' # use those flags, to see which flags are enabled/disabled
    #set_function = WrapSetFunction(lambda idx: compile_run(' '.join(gcc_flags[(1 - idx).astype(bool)]) + ' ' +rest), use_call_dict=True)
    if prog == 'susan':
        set_function = WrapSetFunction(lambda idx: compile_run_susan(' '.join(gcc_flags[(idx).astype(bool)])), use_call_dict=True)
    if prog == 'jpeg':
        set_function = WrapSetFunction(lambda idx: compile_run_jpeg(' '.join(gcc_flags[(idx).astype(bool)])), use_call_dict=True)
    return set_function, n

def load_susan(n=10):
    set_funtion, n = load_prog(prog='susan', n=n)
    return set_funtion, n 

def load_jpeg(n=10):
    set_funtion, n = load_prog(prog='jpeg', n=n)
    return set_funtion, n


@ingredient.named_config
def SUSAN():
    name = 'cbench-automotive-susan'
    set_function, n = load_susan()

@ingredient.named_config
def JPEG():
    name = 'cbench-consumer-jpeg-d'
    set_function, n = load_jpeg()

@ingredient.capture
def get_instance(name, n, set_function, _log):
    """Get an instance of a model according to parameters in the configuration.

    Also, check if the provided parameters fit to the signature of the model
    class and log default values if not defined via the configuration.

    """
    s = set_function
    return s, n
