import numpy as np
import pandas as pd
import itertools

class FitnessFunction:
    def __init__(self, table: pd.DataFrame, cols: tuple, n: int, idx_converter):
        self.n = n
        self.idxcol = cols[0]
        self.valcol = cols[1]
        self.to_idx = idx_converter
        table = table[[self.idxcol, self.valcol]]

        self.f = table.set_index(self.idxcol).T.to_dict(orient='list')
        arr = np.array(list(self.f.values()), dtype=np.float)
        self.max_value = max(np.abs(arr))[0]

    def __call__(self, idx:np.ndarray) -> np.float:
        key = self.to_idx(idx)
        if key in self.f:
            try:
                #print(f'num: {float(self.f[key][0])}, den: {self.max_value}')
                return float(self.f[key][0])
            except:
                #print('what')
                return np.nan
        else:
            return np.nan

    def to_vector(self):
        lst = [list(i)[::-1] for i in itertools.product([0, 1], repeat=self.n)]
        #print(lst)
        vector = [self(np.array(idx)) for idx in lst]
        return vector