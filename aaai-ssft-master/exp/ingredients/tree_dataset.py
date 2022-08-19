import inspect
import pandas as pd

from sacred import Ingredient
import sdsft
from sdsft import treesutils
import numpy as np
from sdsft.common import WrapSetFunction
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

ingredient = Ingredient('dataset')

@ingredient.config
def cfg():
    """Dataset configuration"""
    name = ''
    bins = None
    n_estimators = 100
    max_depth = 4
    test_set = None
    set_function = None
    n = None

def load_supercond(bins, n_estimators, max_depth):
    df = pd.read_csv('./exp/datasets/superconductors/train.csv')
    idxs, crit_temp = treesutils.binarize(df, bins)
    idxs, idxs_test, crit_temp, crit_temp_test = train_test_split(idxs, crit_temp, random_state=0)

    reg = RandomForestRegressor(n_estimators=n_estimators, max_features='log2', max_depth=max_depth, random_state=0)
    reg.fit(idxs, crit_temp)

    s = lambda idx: reg.predict(idx)
    set_function = WrapSetFunction(s)
    n = idxs.shape[1]

    test_set = np.concatenate([idxs_test, crit_temp_test[:, np.newaxis]], axis=1)
    return set_function, n, test_set


@ingredient.named_config
def SUPERCOND():
    name = 'superconductors'
    bins = 5
    n_estimators = 1000
    max_depth = 3
    set_function, n, test_set = load_supercond(bins, n_estimators, max_depth)

@ingredient.capture
def get_bins_no(name, bins, test_set, n, set_function, _log):
    return bins

@ingredient.capture
def get_test(name, bins, test_set, n, set_function, _log):
    return test_set

@ingredient.capture
def get_instance(name, bins, test_set, n, set_function, _log):
    """Get an instance of a model according to parameters in the configuration.

    Also, check if the provided parameters fit to the signature of the model
    class and log default values if not defined via the configuration.

    """
    s = set_function
    return s, n