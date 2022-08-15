import numpy as np
import itertools
import math

class NK_model():
    def __init__(self, N:int, K:int) -> None:
        assert(K>=0 and K<N)
        self.N = N
        self.K = K
        self.neighs = dict()
        self.neighs_eval = dict()
    
    def __call__(self, idx:np.ndarray) -> np.float64:
        #np.random.seed(10)
        assert(len(idx)==self.N)
        val = 0
        for i in range(len(idx)):
            #iterate over each gene i
            if i not in self.neighs:
                #if it doesn't have any choose neighbours at random for i, else use self.neighs to find them
                self.neighs[i] = sorted(np.random.choice([k for k in range(self.N) if k!=i], self.K, replace=False).tolist() + [i])
            i_neighs = self.neighs[i]
            #create key for indexing contribution of gene i to fitness, index=(neighbours of i and i, activation of i)
            key = tuple(zip(i_neighs, idx[i_neighs]))
            if key not in self.neighs_eval:
                self.neighs_eval[key] = np.random.normal(0,10)
            val += self.neighs_eval[key]
        return (val/self.N) + np.random.normal(0,1)
    
    def to_vector(self):
        lst = [list(i) for i in itertools.product([0, 1], repeat=self.N)]
        vector = [self(np.array(idx)) for idx in lst]
        return vector



