from sys import flags
import gym
import compiler_gym
import numpy as np
from sdsft import WrapSetFunction
from sdsft.transforms import SparseSFT
from sdsft.common import eval_sf
from sacred import Ingredient

ingredient = Ingredient('dataset')

@ingredient.config
def cfg():
    """Dataset configuration"""
    name = ''
    set_function = None
    n = None

def set_fun(env, flags, idx):
    choices = -np.ones(len(env.gcc_spec.options),dtype=int)
    choices[flags] = (idx-1)
    env.choices = choices
    print(env.obj_size)
    return env.obj_size

def load_gcc(n):
    np.random.seed(0)
    env = compiler_gym.make("gcc-v0")
    env.reset(benchmark = "generator://csmith-v0/34")#  "benchmark://chstone-v0/sha" "benchmark://anghabench-v1"
    #-Os option flags: -falign-functions  -falign-jumps -falign-labels  -falign-loops -fprefetch-loop-arrays  -freorder-blocks-algorithm=stc
    # indices of options: 2, 3, 4, 5, 116, 125
    flags = list(range(1,n+1))
    print(f'flags indices:\n{flags}')
    set_function = WrapSetFunction(lambda idx: set_fun(env, flags, idx), use_call_dict=True)
    return set_function, n

@ingredient.named_config
def OBJSIZE():
    name = 'obj file size (compiler_gym)'
    set_function, n = load_gcc(80)

@ingredient.capture
def get_instance(name, n, set_function, _log):
    """Get an instance of a model according to parameters in the configuration.

    Also, check if the provided parameters fit to the signature of the model
    class and log default values if not defined via the configuration.

    """
    s = set_function
    return s, n


# set_function, n = load_gcc()
# print(f'loaded')
# SSFT = SparseSFT(n, flag_general=False, flag_print=True, k_max=1000, model='4')
# estimate = SSFT.transform(set_function)

# error = eval_sf(set_function, estimate, n, n_samples=1000)
# print(f'relative error: {error}')