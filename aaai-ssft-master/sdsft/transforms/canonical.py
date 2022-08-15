from functools import reduce
from turtle import shape
import numpy as np
import scipy
import scipy.linalg
import sdsft
from ..common import SetFunction, SparseDSFT4Function, DSFT4OneHop, SparseDSFT3Function, weird_update, DSFT3OneHop
from ..nanutils import clean_nan
from sympy import fwht, ifwht
    
class SparseSFT:
    def __init__(self, n, eps=1e-8, flag_print=False, k_max=None, flag_general=True, model='W3'):
        """
            @param n: ground set size
            @param eps: |x| < eps is treated as zero
            @param flag_print: printing flag
            @param k_max: the maximum amount of frequencies maintained in all steps but the last
            @param flag_general: this toggles the filtering by a random one hop filter and is 
            required to handle arbitrary/adversary Fourier coefficients.
            @param model: model for Fourier transform.
        """
        self.n = n
        self.k_max = k_max
        self.flag_print = flag_print
        self.eps = eps
        self.flag_general = flag_general
        if flag_general:
            self.weights = np.random.normal(0, 1, n)
        self.model = model

    def solve_subproblem(self, s, keys_old, coefs_old, measurements_previous, M_previous):
        n = self.n
        eps = self.eps
        model = self.model
        if self.k_max is None:
            keys_sorted = keys_old
        else:
            cards = keys_old.sum(axis=1)
            mags = -np.abs(coefs_old)
            criteria = np.zeros(len(keys_old), dtype=[('cards', '<i4'), ('coefs', '<f8')])
            criteria['cards'] = cards
            criteria['coefs'] = mags
            idx_order = np.argsort(criteria, order=('cards', 'coefs'))[:self.k_max]
            keys_sorted = keys_old[idx_order]

            M_previous = M_previous[idx_order][:, idx_order]
            measurements_previous = measurements_previous[idx_order]

        n1 = keys_sorted.shape[1]
        measurement_positions = np.zeros((keys_sorted.shape[0], n), dtype=np.int32)
        if (model == 'W3' or model == '3'):
            measurement_positions[:, :n1] = keys_sorted
            measurement_positions[:, n1+1:] = np.ones(n - n1-1, dtype=np.int32)
        if (model == '4'):
            measurement_positions[:, :n1] = 1 - keys_sorted
            measurement_positions[:, n1] = 1
        measurements_new = s(measurement_positions)
        #print(f'measures: {measurements_new}')
        #M_previous, measurements_new, measurements_previous = clean_nan(M_previous, measurements_new, measurements_previous)
        if (model == 'W3'):
            rhs = np.concatenate([measurements_new[:, np.newaxis], 
                                 2./np.sqrt(3) * measurements_previous[:, np.newaxis] - 1./np.sqrt(3) * measurements_new[:, np.newaxis]],
                                axis=1)
        if (model == '3'):
            rhs = np.concatenate([measurements_new[:, np.newaxis],
                                 measurements_new[:, np.newaxis] - measurements_previous[:, np.newaxis]],
                                axis=1)
        if (model == '4'):
            rhs = np.concatenate([measurements_new[:, np.newaxis],
                                 measurements_previous[:, np.newaxis] - measurements_new[:, np.newaxis]],
                                axis=1)
        coefs = scipy.linalg.solve_triangular(M_previous, rhs, lower=True)
        n_queries = len(measurements_new)
        # if(n1<self.n-1):
        #     support_first = np.where(np.abs(coefs[:,0]) + np.abs(coefs[:,1]) > eps)[0]
        #     support_second = np.where(np.abs(coefs[:,0]) + np.abs(coefs[:,1]) > eps)[0]
        # else:
        support_first = np.where(np.abs(coefs[:, 0]) > eps)[0]
        support_second = np.where(np.abs(coefs[:, 1]) > eps)[0]
        dim1 = len(support_first)
        dim2 = len(support_second)
        dim = len(support_first) + len(support_second)
        M = np.zeros((dim, dim), dtype=np.float32)
        if (model == 'W3'):
            M[:dim1, :dim1] = M_previous[support_first][:, support_first]
            M[dim1:, :dim1] = 0.5 * M_previous[support_second][:, support_first]
            M[dim1:, dim1:] = 0.5 * np.sqrt(3) * M_previous[support_second][:, support_second]
            # Mtest = 0.5 * np.sqrt(3) * M_previous[support_second][:, support_second]
            # print(f'correct:\n {Mtest}')
            # M[dim1:, dim1:] =  weird_update(M[dim1:, dim1:], n1, support_second, measurement_positions)
            # print(f'not correct:\n {M[dim1:, dim1:]}')
        if(model == '3'):
            M[:dim1, :dim1] = M_previous[support_first][:, support_first]
            M[dim1:, :dim1] = M_previous[support_second][:, support_first]
            M[dim1:, dim1:] = -M_previous[support_second][:, support_second]
        if(model == '4'):
            M[:dim1, :dim1] = M_previous[support_first][:, support_first]
            M[dim1:, :dim1] = M_previous[support_second][:, support_first]
            M[dim1:, dim1:] = M_previous[support_second][:, support_second]
        measurements = np.concatenate([measurements_new[support_first], measurements_previous[support_second]])
        if(model == 'W3' or model == '3'):
            keys_first = measurement_positions[support_first][:, :n1 + 1]
            keys_second = measurement_positions[support_second][:, :n1 + 1]
        if(model == '4'):
            keys_first = 1 - measurement_positions[support_first][:, :n1 + 1]
            keys_second = 1 - measurement_positions[support_second][:, :n1 + 1]
        keys_second[:, -1] = 1
        keys = np.concatenate([keys_first, keys_second], axis=0)
        fourier_coefs = np.concatenate([coefs[support_first][:, 0], coefs[support_second][:, 1]])
        return fourier_coefs, keys, measurements, M, n_queries

    def transform(self, X0):
        n = self.n
        model = self.model
        if self.flag_general:
            if model == '4':
                s = DSFT4OneHop(n, self.weights, X0)
                self.hs = s
            if model == '3':
                s = DSFT3OneHop(n, self.weights, X0)
                self.hs = s
        else:
            s = X0
        if(model == 'W3' or model == '3'):
            sN = s(np.ones(n, dtype=np.bool))[0]
        if(model == '4'):
            sN = s(np.zeros(n, dtype=np.bool))[0]
        M = np.ones((1, 1), dtype=np.float64)
        keys = np.zeros((1, 0), dtype=np.int32)
        fourier_coefs = np.ones(1, dtype=np.float64)*sN
        measurements = np.ones(1, dtype=np.float64)*sN
        partition_dict = {():sN}
            
        n_queries_total = 0
        for k in range(n):
            if len(list(partition_dict.keys())) == 0:
                keys = np.zeros((1, n), dtype=np.int32)
                fourier_coefs = np.zeros(1, dtype=np.float64)
                break
            try:
                fourier_coefs, keys, measurements, M, n_queries = self.solve_subproblem(s, keys, fourier_coefs, measurements, M)
            except ValueError as e:
                partition_dict = {}
            
            if self.flag_print:
                print('iteration %d: queries %d'%(k+1, n_queries))
            n_queries_total += n_queries

        if self.flag_print:
            print('total queries: %d'%n_queries_total)
        if(model == 'W3' or model == '3'):
            estimate = SparseDSFT3Function(keys, fourier_coefs, model = model)
        if(model == '4'):
            estimate = SparseDSFT4Function(keys, fourier_coefs)

        if self.flag_general:
            estimate = s.convertCoefs(estimate)
        
        return estimate




    
    
    
    
    
    
    
