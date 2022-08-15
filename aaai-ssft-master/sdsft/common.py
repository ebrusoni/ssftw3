from abc import ABC, abstractmethod
from matplotlib.pyplot import axis
import numpy as np
import itertools


class SetFunction(ABC):    
    def __call__(self, indicators, count_flag=True):
        """
            @param indicators: two dimensional np.array of type np.int32 or np.bool
            with one indicator vector per row
            @param count_flag: a flag indicating whether to count set function evaluations
            @returns: a np.array of set function evaluations
        """
        pass

class WrapSetFunction(SetFunction):
    def __init__(self, s, use_call_dict=False, use_loop=False):
        self.s = s
        self.call_counter = 0
        self.use_loop = use_loop
        if use_call_dict:
            self.call_dict = {}
        else:
            self.call_dict = None
        
        
    def __call__(self, indicator, count_flag=True):
        if len(indicator.shape) < 2:
            indicator = indicator[np.newaxis, :]
        
        result = []
        if self.call_dict is not None:
            for ind in indicator:
                key = tuple(ind.tolist())
                if key not in self.call_dict:
                    self.call_dict[key] = self.s(ind)
                    if count_flag:
                        self.call_counter += 1
                result += [self.call_dict[key]]
            return np.asarray(result)
        elif self.use_loop:
            result = []
            for ind in indicator:
                result += [self.s(ind)]
                if count_flag:
                    self.call_counter += 1
            return np.asarray(result)
        else:
            if count_flag:
                self.call_counter += indicator.shape[0]
            return self.s(indicator)

    def to_vector(self, n):
        lst = [list(i)[::-1] for i in itertools.product([0, 1], repeat=n)]
        #print(lst)
        vector = [self.s(np.array(idx)) for idx in lst]
        return vector


class SparseDSFT4Function(SetFunction):
    
    def __init__(self, frequencies, coefficients):
        """
            @param frequencies: two dimensional np.array of type np.int32 or np.bool 
            with one indicator vector per row
            @param coefficients: one dimensional np.array of corresponding Fourier 
            coeffients
        """
        self.freqs = frequencies
        self.coefs = coefficients
        self.call_counter = 0
        
        
    def __call__(self, indicators, count_flag=True):
        """
            @param indicators: two dimensional np.array of type np.int32 or np.bool
            with one indicator vector per row
            @param count_flag: a flag indicating whether to count set function evaluations
            @returns: a np.array of set function evaluations
        """
        ind = indicators
        freqs = self.freqs
        coefs = self.coefs
        if len(ind.shape) < 2:
            ind = ind[np.newaxis, :]
        active = freqs.dot(ind.T)
        active = active == 0
        res = (active * coefs[:, np.newaxis]).sum(axis=0)
        return res

class SparseDSFT3Function(SetFunction):
    
    def __init__(self, frequencies, coefficients, model='3'):
        """
            @param frequencies: two dimensional np.array of type np.int32 or np.bool 
            with one indicator vector per row
            @param coefficients: one dimensional np.array of corresponding Fourier 
            coeffients
        """
        self.freq_sums = frequencies.sum(axis=1)
        self.freqs = frequencies
        self.coefs = coefficients
        self.call_counter = 0
        self.model = model
        
        
    def __call__(self, indicators, count_flag=True):
        """
            @param indicators: two dimensional np.array of type np.int32 or np.bool
            with one indicator vector per row
            @param count_flag: a flag indicating whether to count set function evaluations
            @returns: a np.array of set function evaluations
        """
        ind = indicators
        freqs = self.freqs
        coefs = self.coefs
        fsum = self.freq_sums
        if len(ind.shape) < 2:
            ind = ind[np.newaxis, :]
        if(self.model == '3'):
            coefs = (-1)**freqs.sum(axis=1) * coefs
            active = freqs.dot(np.logical_not(ind).T)
            active = active == 0
        if(self.model == 'W3'):
            coefs = (np.sqrt(3))**(fsum) * coefs
            active = freqs.dot(np.logical_not(ind).T)
            active = (active == 0) * (0.5**ind.sum(axis=1))
        res = (active * coefs[:, np.newaxis]).sum(axis=0)
        return res

class ReverseSetFunction(SetFunction):
    def __init__(self, s):
        self.s = s
        self.call_counter = 0
        
    def __call__(self, indicator, count_flag=True):
        if len(indicator.shape) < 2:
            indicator = indicator[np.newaxis, :]
        return self.s(1 - indicator)


def int2indicator(A, n_groundset):
    indicator = [int(b) for b in bin(2**n_groundset + A)[3:][::-1]]
    indicator = np.asarray(indicator, dtype=np.int32)
    return indicator

class DSFT3OneHop(SetFunction):
    
    def __init__(self, n, weights, set_function):
        self.n = n
        self.weights = weights
        self.s = set_function
        self.call_counter = 0
    
    def __call__(self, indicators, count_flag=True ):
        if len(indicators.shape) < 2:
            indicators = indicators[np.newaxis, :]
        
        s = self.s
        weights = self.weights
        res = []
        for ind in indicators:
            nc = np.sum(ind)
            if count_flag:
                self.call_counter += (nc + 1)
            mask = ind.astype(np.int32)==1
            ind_shifted = np.tile(ind, [nc, 1])
            ind_shifted[:, mask] = 1-np.eye(nc, dtype=ind.dtype)
            ind_one_hop = np.concatenate((ind[np.newaxis], ind_shifted), axis=0)
            weight_s0 = np.ones(1)*(1 + weights[True^mask].sum())
            active_weights = np.concatenate([weight_s0, weights[mask]])
            res += [(s(ind_one_hop)*active_weights).sum()]
        res = np.asarray(res)
        return res
    
    def convertCoefs(self, estimate):
        freqs = estimate.freqs
        coefs = estimate.coefs
        coefs_new = []
        freqs = freqs.astype(bool)
        for key, value in zip(freqs, coefs):
            coefs_new += [value/(1 + self.weights[True^key].sum())]
        return SparseDSFT3Function(freqs.astype(np.int32), np.asarray(coefs_new))
    
class DSFT4OneHop(SetFunction):
    
    def __init__(self, n, weights, set_function):
        self.n = n
        self.weights = weights
        self.s = set_function
        self.call_counter = 0
    
    def __call__(self, indicators, count_flag=True, sample_optimal=True):
        if len(indicators.shape) < 2:
            indicators = indicators[np.newaxis, :]
        
        s = self.s
        weights = self.weights
        if sample_optimal:
            res = []
            for ind in indicators:
                nc = ind.shape[0]-np.sum(ind)
                if count_flag:
                    self.call_counter += (nc + 1)
                mask = ind.astype(np.int32)==0
                ind_shifted = np.tile(ind, [nc, 1])
                ind_shifted[:, mask] = np.eye(nc, dtype=ind.dtype)
                ind_one_hop = np.concatenate((ind[np.newaxis], ind_shifted), axis=0)
                weight_s0 = np.ones(1)*(1 + weights[True^mask].sum())
                active_weights = np.concatenate([weight_s0, weights[mask]])
                res += [(s(ind_one_hop)*active_weights).sum()]
            res = np.asarray(res)
        else:
            res = s(indicators)
            for i, weight in enumerate(weights):
                ind_shifted = indicators.copy()
                ind_shifted[:, i] = 1
                res += weight*s(ind_shifted)
            if count_flag:
                self.call_counter += (self.n + 1) * indicators.shape[0]
        return res
    
    def convertCoefs(self, estimate):
        freqs = estimate.freqs
        coefs = estimate.coefs
        coefs_new = []
        freqs = freqs.astype(np.bool)
        for key, value in zip(freqs, coefs):
            coefs_new += [value/(1 + self.weights[True^key].sum())]
        return SparseDSFT4Function(freqs.astype(np.int32), np.asarray(coefs_new))

class SparseWHTFunction(SetFunction):
    
    def __init__(self, frequencies, coefficients, normalization=False):
        """
            @param frequencies: two dimensional np.array of type np.int32 with 
            one indicator vector per row. Important: int and not bool!
            @param coefficients: one dimensional np.array of corresponding Fourier 
            coeffients
        """
        self.freqs = frequencies.astype(np.int32)
        self.coefs = coefficients
        self.call_counter = 0
        self.normalization = normalization
        
    def __call__(self, indicators, count_flag=True):
        """
            @param indicators: two dimensional np.array of type np.int32 or np.bool
            with one indicator vector per row
            @param count_flag: a flag indicating whether to count set function evaluations
            @returns: a np.array of set function evaluations
        """    
        ind = indicators.astype(np.int32)
        freqs = self.freqs
        coefs = self.coefs
        if len(ind.shape) < 2:
            ind = ind[np.newaxis, :]
        n = freqs.shape[1]
        factor = 1
        if self.normalization:
            factor = (1/2)**n
        A_cap_B = freqs.dot(ind.T)

        res = factor*((-1)**A_cap_B * coefs[:, np.newaxis]).sum(axis=0)
        return res

def eval_sf(gt, estimate, n, n_samples=1000, err_type="rel", custom_samples=None, p=0.5):
    """
        @param gt: a SetFunction representing the ground truth
        @param estimate: a SetFunction 
        @param n: the size of the ground set
        @param n_samples: number of random measurements for the evaluation
        @param err_type: either mae or relative reconstruction error
    """
    if custom_samples is None:
        ind = np.random.binomial(1, p, (n_samples, n)).astype(np.bool)
    else:
        ind = custom_samples
    gt_vec = gt(ind, count_flag=False)
    est_vec = estimate(ind, count_flag=False)

    #for dealing with nan in set function
    # nan_entries = np.where(np.isnan(gt_vec))
    # if(len(nan_entries[0])>0):
    #     print(nan_entries)
    # gt_vec = np.delete(gt_vec, nan_entries, axis=0)
    # est_vec = np.delete(est_vec, nan_entries, axis=0)
    
    if err_type=="mae":
        return (np.linalg.norm(gt_vec - est_vec, 1)/n_samples)
    elif err_type=="rel":
        return np.linalg.norm(gt_vec - est_vec)/np.linalg.norm(gt_vec)
    elif err_type=="inf":
        return np.linalg.norm(gt_vec - est_vec, ord=np.inf)
    elif err_type=="res_quantiles":
        return np.quantile(np.abs(gt_vec - est_vec), [0.25, 0.5, 0.75])
    elif err_type=="quantiles":
        return np.quantile(np.abs(gt_vec), [0.25, 0.5, 0.75])
    elif err_type=="res":
        return gt_vec - est_vec
    elif err_type=="raw":
        return gt_vec, est_vec
    elif err_type=="R2":
        gt_mean = np.mean(gt_vec)
        return 1 - np.mean((est_vec - gt_vec)**2)/np.mean((gt_vec - gt_mean)**2)
    else:
        raise NotImplementedError("Supported error types: mae, rel, inf, res_quantiles, quantiles")


def gains(s, N, S0):
    max_value = -np.inf
    max_el = -1
    for element in N[True^S0]:
        curr_indicator = S0.copy()
        curr_indicator[element] = True
        curr_value = s(curr_indicator, count_flag=False)[0]
        if curr_value > max_value:
            max_value = curr_value
            max_el = element
        elif curr_value == max_value:
            if np.random.rand() > 0.5:
                max_value = curr_value
                max_el = element
    return max_el, max_value

def maximize_greedy(s, n, card, verbose=False, force_card=False):
    S0 = np.zeros(n, dtype=np.bool)
    N = np.arange(n)
    for t in range(card):
        i, value = gains(s, N, S0)
        if verbose:
            print('gains: i=%d, value=%.4f'%(i, value))
            print(S0.astype(np.int32))
        if value > 0 or force_card:
            S0[i] = 1
        else:
            break
    return S0, s(S0, count_flag=False)[0]

def weird_update(M, n1, support_second, measurement_positions):
    for (i, idx_i) in enumerate(support_second):
        A = np.sum(measurement_positions[idx_i, :n1])
        for (j, idx_j) in enumerate(support_second):
            if not contains(measurement_positions[idx_i, :n1], measurement_positions[idx_j, :n1]):
                continue
            B = np.sum(measurement_positions[idx_j, :n1])
            if (B % 2 == 0):
                M[i][j] = 2.**(-A-1) * ((3.**(B/2)) * np.sqrt(3))
            else:
                M[i][j] = 2.**(-A-1) * (3.**((B+1)/2))
    return M

def contains(ind_A, ind_B):
    active = ind_B.dot(np.logical_not(ind_A).T)
    return active == 0
