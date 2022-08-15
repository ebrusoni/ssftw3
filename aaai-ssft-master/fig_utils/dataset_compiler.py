from sys import flags
import gym
import compiler_gym
import numpy as np
from sdsft import WrapSetFunction
from sdsft.transforms import SparseSFT
from sdsft.common import eval_sf
import time

def set_fun(env, flags, idx):
    choices = -np.ones(len(env.gcc_spec.options),dtype=int)
    choices[flags] = (idx-1)
    env.choices = choices
    print(env.obj_size)
    return env.obj_size

def load_gcc():
    np.random.seed(0)
    env = compiler_gym.make("gcc-v0")
    #print(list(env.datasets))
    env.reset(benchmark = "generator://csmith-v0/34")#"benchmark://chstone-v0/sha"
    #-Os option flags: -falign-functions  -falign-jumps -falign-labels  -falign-loops -fprefetch-loop-arrays  -freorder-blocks-algorithm=stc
    # indices of options: 2, 3, 4, 5, 116, 125
    #print(list(enumerate(env.gcc_spec.options)))
    n = 5#len(env.gcc_spec.options)
    #flags = np.sort(np.random.choice(len(env.gcc_spec.options), n, replace=False))
    # print(flags)
    flags = list(range(1,n+1))
    # flags = np.array([2, 3, 4, 5, 116, 125])
    # print(env.gcc_spec.options[116])
    print(f'flags indices:\n{flags}')
    set_function = WrapSetFunction(lambda idx: set_fun(env, flags, idx), use_call_dict=True)
    return set_function, n

set_function, n = load_gcc()
print(f'loaded')
SSFT = SparseSFT(n, flag_general=False, flag_print=True, k_max=1000, model='4')
start = time.time()
estimate = SSFT.transform(set_function)
end = time.time()

error = eval_sf(set_function, estimate, n, n_samples=1000)
print(f'relative error: {error}, time: {end-start}')


