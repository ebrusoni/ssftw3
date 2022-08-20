import sys
sys.path.append('..')
import numpy as np
import pandas as pd
#from ssft3_test import random_s

import sdsft.transforms.canonical as can
from sdsft.transforms import SparseSFT
import sdsft.common as com
from exp.ingredients import fitness_dataset as dataset
from exp.ingredients import exectime_dataset as ck_dataset
from sympy import fwht
import matplotlib.pyplot as plt
import itertools
from fig_utils.bin_plot import get_avg_coefs

'''
Python file for performing DSFT3, DSFT4, DSFTW3 and WHT by matrix multiplication.
Generates the binned plots presented in the manuscript.

To load a specific set function just replace the line below with the set function you wish.
The set function loaders (load_<dataset>()) can be found in ../exp/ingredients

'''

#set_function,n = ck_dataset.load_susan()

def scale_order(coefs, order=None):
    if order is not None:
        coefs = np.abs(coefs[order])
    else:
        coefs = np.abs(coefs)
    scaled_coefs = (coefs - np.min(coefs)) / (np.max(coefs) - np.min(coefs))
    return scaled_coefs

def plot_coefs(coefs_label_list, n):
    fig, axes = plt.subplots(4,1)
    i = 0
    colors = ['tab:red', 'tab:orange', 'tab:blue', 'tab:green']
    for coefs, l in coefs_label_list:
        axes[i].plot(range(0, 2**n), coefs, colors[i], label=l)
        axes[i].set(ylabel=l+'\ncoeffs')
        i+=1
    for ax in axes.flat:
        ax.set(xlabel='frequencies')
    for ax in axes.flat:
        ax.label_outer()

    #plt.savefig('test', format='pdf')
    plt.show()
    plt.close()

def plot_bincoefs(coefs_label_list):
    n = len(coefs_label_list[0][0])
    fig, axes = plt.subplots(1,1)

    i = 0
    colors = ['tab:red', 'tab:orange', 'tab:blue', 'tab:green']
    for avg_coefs, l in coefs_label_list:
        axes.plot(avg_coefs, label=l)
    axes.set(ylabel='Avg. Coefficient')
    axes.set(xlabel='Cardinality of Frequency')
    plt.legend()
    #plt.savefig('test_avg', format='pdf')
    plt.show()
    plt.close()


def plot(set_function, n, centre=False):
    MW3 = np.array([[1               , 0],
                    [-1. / np.sqrt(3), 2. / np.sqrt(3)]])
    M3 = np.array([[1, 0],
                   [1,-1]])
    M4 = np.array([[0, 1],
                   [1,-1]])
    
    TW3 = MW3
    T3 = M3
    T4 = M4
    
    for i in range(n-1):
        TW3 = np.kron(TW3, MW3)
        T3  = np.kron(T3, M3)
        T4  = np.kron(T4, M4)
    
    s = np.array(set_function.to_vector(n)).reshape(2**n)
    if centre:
        s = s - np.mean(s)
    print(f's: {s}')
    inds = np.array([list(i)[::-1] for i in itertools.product([0, 1], repeat=n)])
    
    mags = np.sum(inds, axis=1)
    sorted_inds = np.argsort(mags)
    
    #ssftw3 = SparseSFT(n, eps=1e-10, model='W3', flag_general=False)
    #print(f'SSFTW3: {ssftw3.transform(set_function).coefs}')
    coeffs_w3 = np.matmul(TW3, s)
    #print(f'matmul: {coeffs_w3}')
    ftw3 = com.SparseDSFT3Function(inds, coeffs_w3, model = 'W3')
    
    coeffs_3 = np.matmul(T3, s)
    ft3 = com.SparseDSFT3Function(inds, coeffs_3, model = '3')
    
    coeffs_4 = np.matmul(T4, s)
    ft4 = com.SparseDSFT4Function(inds, coeffs_4)
    
    coeffs_wht = np.array(fwht(s), dtype=np.float32)
    ftwht = com.SparseWHTFunction(inds, coeffs_wht)
    
    coefs_labels = [(scale_order(coeffs_w3, sorted_inds), 'WDSFT3'),
                    (scale_order(coeffs_3, sorted_inds), 'DSFT3'),
                    (scale_order(coeffs_4, sorted_inds), 'DSFT4'),
                    (scale_order(coeffs_wht, sorted_inds), 'WHT')
                    ]
    
    #plot_coefs(coefs_labels, n)
    avg_coefs_w3 = get_avg_coefs(ftw3, n, flag_rescale=True)
    avg_coefs_3 = get_avg_coefs(ft3, n, flag_rescale=True)
    avg_coefs_4 = get_avg_coefs(ft4, n, flag_rescale=True)
    avg_coefs_wht = get_avg_coefs(ftwht, n, flag_rescale=True)
    
    avg_coefs_labels = [(avg_coefs_w3, 'WDSFT3'),
                    (avg_coefs_3, 'DSFT3'),
                    (avg_coefs_4, 'DSFT4'),
                    (avg_coefs_wht, 'WHT')
                    ]
    
    plot_bincoefs(avg_coefs_labels)
    
    # print(f'W3 coefs: {coeffs_w3}')
    # print(f'W3 avg: {avg_coefs_w3}')
    # print(f'3: {avg_coefs_3}')
    # print(f'4: {avg_coefs_4}')
    # print(f'wht: {avg_coefs_wht}')

if __name__=="__main__":
    set_function, n = dataset.load_devisser()
    plot(set_function, n, centre=False)