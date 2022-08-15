from ast import Lambda
import pandas as pd
import numpy as np
import math

def find_idx(test_bin: np.array, idx: np.array):
    #if there are multiple results return the first one
    idx_temp = np.where((test_bin == idx).all(axis=1))[0][0]
    return idx_temp

def get_stats(gt_vec: np.array, est_vec: np.array):
    rel = np.linalg.norm(gt_vec - est_vec)/np.linalg.norm(gt_vec)
    mae = np.mean(np.abs(gt_vec - est_vec))
    inf = np.linalg.norm(gt_vec - est_vec, ord=np.inf) 
    return (rel, mae, inf)

def one_hot_enc(bin: int, bins: int):
    enc = np.zeros(bins, dtype=np.int)
    enc[bin] = 1
    return enc

def determine_bin(entry:float, range: tuple, bins:int) -> np.array:
    #print(f'entry: {entry}, range: {range}, bins {bins}')
    low, high = range
    bin_size = (high - low) / bins
    bin = min(bins - 1, math.floor((entry - low) / bin_size))
    #print(f'bin: {bin}')
    return bin

def get_range(df_column: pd.DataFrame):
    return (df_column.min(), df_column.max())

def binarize(df: pd.DataFrame, bins: int):
    for col in df.columns[:-1]:
        range_col = get_range(df[col])
        f = lambda entry : determine_bin(entry, range_col, bins)
        df[col] = df[col].apply(f)

    arr = df.iloc[:,:-1].to_numpy()
    y = df.iloc[:,-1].to_numpy(dtype=np.int)   
    samples, feats = arr.shape
    arr = arr.flatten()
    #print(arr)

    vect_arr = np.zeros((samples * feats, bins))
    vect_arr[np.arange(samples * feats), arr] = 1
    vect_arr = vect_arr.reshape((samples, feats * bins))
    
    return vect_arr, y
