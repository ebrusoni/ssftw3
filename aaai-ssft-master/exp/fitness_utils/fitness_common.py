import numpy as np
import pandas as pd
from .FitnessFunction import FitnessFunction as FitFun

def wrap_dataset(Ind, Val, n: int, get_idx):
    IndVal = pd.concat([Ind, Val], axis=1)
    s = FitFun(IndVal, IndVal.columns, n, get_idx)
    return s

def idx_to_string(idx: np.ndarray) -> str:
    return str(idx)

def remove_from_str(word: str, strings: list) -> str:
    for s in strings:
        word = word.replace(s, '')
    return word

def merge_idx_cols(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    df = df.iloc[:, start:end]
    arr = df.to_numpy(dtype=int)
    arr = np.array([''.join(row) for row in arr.astype(str)])
    return pd.DataFrame(arr)

def get_idx_from_elements(ground_set: list, idx: np.ndarray):
    if np.sum(idx)==0:
        return 'anc'
    n = len(ground_set)
    map_el_idx = dict(zip(range(0, n), ground_set))
    strIdx = ""
    for i in range(idx.shape[0]):
        if idx[i] == 1:
            strIdx += map_el_idx[i] 
    return strIdx

def get_idx_RED_BLUE(idx: np.ndarray) -> str:
    idx = idx.astype(int)
    tr = ''.maketrans('', '', '[], ')
    strIdx = '\''+str(idx).translate(tr)+'\''
    return strIdx

def get_idx_RTSGP(idx: np.ndarray) -> str:
    ground_set = ['r', 't', 's', 'g', 'p']
    return get_idx_from_elements(ground_set, idx)

def get_idx_FIRST(idx: np.ndarray) -> str:
    index = remove_from_str(str(idx.astype(int)), ['[', ']', ',', ' '])
    return str(index)

def get_idx_genopheno(idx: np.array) -> np.array:
    index = 0
    idx= idx[::-1].astype(int)
    index = remove_from_str(str(idx), ['[', ']', ',', ' ', '\''])
    return index

def get_idx_franke(idx: np.array) -> np.array:
    index = remove_from_str(str(idx.astype(int)), ['[', ']', ',', ' '])
    return index

def get_idx_common(idx: np.array) -> np.array:
    index = remove_from_str(str(idx.astype(int)), ['[', ']', ',', ' '])
    return index

def permute_fs(s, permutation):
    s_perm = lambda idx: s(idx[list(permutation)])
    return s_perm


