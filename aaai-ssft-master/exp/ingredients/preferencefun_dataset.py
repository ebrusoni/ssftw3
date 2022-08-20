import inspect
from os import read

from sacred import Ingredient

import functools

import sdsft
import numpy as np
from sdsft.common import WrapSetFunction
import scipy.io

ingredient = Ingredient('dataset')

def pref_func(idx, network):
    if np.any(idx):
        maxs = np.max(network[idx.astype(np.bool)], axis=0)
        assert(maxs.shape == (3424,))
        return np.sum(maxs)
    else:
        return 0.0

@ingredient.config
def cfg():
    """Dataset configuration."""
    name = ''
    set_function = None
    n = None

def load_water_impurity(n=20):
    mat = scipy.io.loadmat('./exp/datasets/preference_function/water_imp1000.mat')
    arr = mat['Z1'].toarray()
    maxs = np.max(arr, axis=1)
    idxs = np.argsort(-maxs)

    network = arr[idxs[:n]]
    set_function = WrapSetFunction(lambda idx : pref_func(idx, network), use_loop=True)

    return set_function, n

@ingredient.named_config
def LESKOVEC():
    name = 'leskovec 2007'
    n = 20
    set_function, n = load_water_impurity(n)

@ingredient.capture
def get_instance(name, n, set_function, _log):
    """Get an instance of a model according to parameters in the configuration.

    Also, check if the provided parameters fit to the signature of the model
    class and log default values if not defined via the configuration.

    """
    s = set_function
    return s, n