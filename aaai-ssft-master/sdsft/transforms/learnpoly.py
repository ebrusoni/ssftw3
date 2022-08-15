#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 15:06:10 2022

@author: chrisw
"""

import numpy as np
from ..common import SparseDSFT3Function, SparseDSFT4Function, \
    SetFunction, int2indicator, ReverseSetFunction
import scipy
import tqdm
import sys
from collections import defaultdict

class LearnPolyDSFT4():
    """
    Use LearnPoly, which computes the DSFT3, to compute DSFT4.
    """
    def __init__(self, n, tres=1e-6, equivalence_budget=100000, total_budget=1000000, record_error=False, flag_print=False, flag_refit=False):
        self.record_error = record_error
        self.lp = LearnPoly(n, tres=tres, equivalence_budget=equivalence_budget, total_budget=total_budget, record_error=record_error, flag_print=flag_print, flag_refit=flag_refit)
        
    def transform(self, s):
        r = ReverseSetFunction(s)
        dsft3 = self.lp.transform(r)
        return SparseDSFT4Function(dsft3.freqs, dsft3.coefs)
    
    def getStats(self):
        if self.record_error:
            return self.lp.stats
        else:
            return None

class LearnPoly():
    """
    Implementation of "Sparse Multivariate Polynomials over a Field with Queries and Counterexamples", 
    https://www.sciencedirect.com/science/article/pii/S0022000096900173
    """
    def __init__(self, n, tres=1e-6, equivalence_budget=100000, total_budget=1000000, record_error=False, flag_print=False, flag_refit=False, flag_efficient_hyp=True):
        self.n = n
        self.tres = tres
        self.print_flag = flag_print
        self.equivalence_budget = equivalence_budget
        self.total_budget = total_budget
        self.measurements = {}
        self.S = {}
        self.est = None
        self.n_measurements_last = 0
        self.flag_refit = flag_refit
        self.record_error = record_error
        self.n_queries = 0
        self.flag_efficient_hyp = flag_efficient_hyp
        if flag_efficient_hyp:
            self.hyp = self.hyp_cheaper_solves
        else:
            self.hyp = self.hyp_naive
        
    def _inv_DSFT3(self, rows, columns):
        Finv = (np.dot(rows, columns.T) == columns.sum(axis=1)).astype(np.int32)
        return Finv
        
    def hyp_naive(self):
        if self.est is None or self.n_measurements_last < len(list(self.S.values())):
            if self.print_flag:
                print('hyp called on a sample of %d'%len(self.S))
                print('measurments so far %d'%len(self.measurements)) 
                
            inds = np.asarray(list(self.S.keys()), dtype=np.int32)
            vals = np.asarray(list(self.S.values()))
            
            indices = np.lexsort(inds.T)
            inds = inds[indices]
            vals = vals[indices]
            
            freqs = inds
            if self.print_flag:
                print('solve...')
            X = self._inv_DSFT3(inds, freqs)
            #print(X)
            #print((X == np.tril(X)).all())
            
            #coefs = np.linalg.solve(X.T.dot(X) + 0*np.eye(len(inds)), X.T.dot(vals))
            
            coefs = scipy.linalg.solve_triangular(X, vals, trans=0, lower=True, unit_diagonal=True, overwrite_b=True, debug=None, check_finite=True)
            
            if self.print_flag:
                print('post solve...')# can be speedup with if the freqs have the right order, scipy.linalg.solve_triangular(M_previous, rhs, lower=True)        freqs = freqs[np.abs(coefs) > self.tres]
            freqs = freqs[np.abs(coefs) > self.tres]
            coefs = coefs[np.abs(coefs) > self.tres]
            self.n_measurements_last = len(vals)
            self.est = SparseDSFT3Function(freqs, coefs)

        return self.est
    
    def hyp_cheaper_solves(self):
        if self.est is None:
            inds = np.asarray(list(self.S.keys()), dtype=np.int32)
            vals = np.asarray(list(self.S.values()))
            
            indices = np.lexsort(inds.T)
            inds = inds[indices]
            vals = vals[indices]
            freqs = inds
            X = self._inv_DSFT3(inds, freqs)
            coefs = scipy.linalg.solve_triangular(X, vals, trans=0, lower=True, unit_diagonal=True, overwrite_b=True, debug=None, check_finite=True)
            self.est = SparseDSFT3Function(freqs, coefs)
            self.Snew = {}
            return self.est
        elif len(self.Snew) == 0:
            return self.est
        elif len(self.Snew) == 1:
            
            
            est = self.est
            est_dict = {tuple(freq.tolist()):coef for freq, coef in zip(est.freqs, est.coefs)}
            
            inds_all = np.asarray(list(self.S.keys()), dtype=np.int32)
            vals_all = np.asarray(list(self.S.values()))
            
            """
            leq_mask = np.zeros(inds_all.shape[0], dtype=bool)
            for key in self.Snew.keys():
                arr = np.asarray(key)
                leq_mask |= inds_all.dot(arr) == arr.sum()
                print(leq_mask.astype(np.int32))
            inds = inds_all[leq_mask]
            vals = vals_all[leq_mask]
            indices = np.lexsort(inds.T)
            inds = inds[indices]
            vals = vals[indices] - est(inds)
            freqs = inds
            X = self._inv_DSFT3(inds, freqs)
            coefs = scipy.linalg.solve_triangular(X, vals, trans=0, lower=True, unit_diagonal=True, overwrite_b=True, debug=None, check_finite=True)
            new_dict = defaultdict(float)
            for freq, coef in zip(freqs, coefs):
                fkey = tuple(freq.tolist())
                if np.abs(coef) > self.tres:
                    new_dict[fkey] += coef
                if fkey in est_dict:
                    new_dict[fkey] += est_dict[fkey]
  
            est_dict = new_dict
            """
            for key in self.Snew.keys():
                new_dict = defaultdict(float)
                for ekey, val in est_dict.items():
                    new_dict[ekey] = val
                arr = np.asarray(key)
                leq_mask = inds_all.dot(arr) == arr.sum()
                
                inds = inds_all[leq_mask]
                vals = vals_all[leq_mask]
 
                indices = np.lexsort(inds.T)
                inds = inds[indices]
                vals = vals[indices] - est(inds)
                freqs = inds
                X = self._inv_DSFT3(inds, freqs)
            
                coefs = scipy.linalg.solve_triangular(X, vals, trans=0, lower=True, unit_diagonal=True, overwrite_b=True, debug=None, check_finite=True)
                for freq, coef in zip(freqs, coefs):
                    fkey = tuple(freq.tolist())
                    if len(new_dict)==0 or np.abs(coef) > self.tres:
                        new_dict[fkey] += coef
                est_dict = new_dict
                est = SparseDSFT3Function(np.asarray(list(est_dict.keys()), dtype=np.int32), np.asarray(list(est_dict.values())))
                
            freqs = np.asarray(list(est_dict.keys()), dtype=np.int32)
            coefs = np.asarray(list(est_dict.values()))
            self.Snew = {}
            self.est = SparseDSFT3Function(freqs, coefs)
            return self.est
        else:
            inds = np.asarray(list(self.S.keys()), dtype=np.int32)
            vals = np.asarray(list(self.S.values()))
            
            indices = np.lexsort(inds.T)
            inds = inds[indices]
            vals = vals[indices]
            freqs = inds
            X = self._inv_DSFT3(inds, freqs)

            
            coefs = scipy.linalg.solve_triangular(X, vals, trans=0, lower=True, unit_diagonal=True, overwrite_b=True, debug=None, check_finite=True)
            
            if self.print_flag:
                print('post solve...')# can be speedup with if the freqs have the right order, scipy.linalg.solve_triangular(M_previous, rhs, lower=True)        freqs = freqs[np.abs(coefs) > self.tres]
            freqs = freqs[np.abs(coefs) > self.tres]
            coefs = coefs[np.abs(coefs) > self.tres]
            self.n_measurements_last = len(vals)
            self.est = SparseDSFT3Function(freqs, coefs)
            self.Snew = {}
            return self.est

    def easyCounterExample(self, s):
        T = []
        s_est = self.hyp()
        freqs = s_est.freqs
        while len(T) < len(freqs):
            f_set = set(tuple(f.tolist()) for f in freqs)
            t_set = set(tuple(t.tolist()) for t in T)
            freqs_minus_T = np.asarray(list(f_set - t_set), dtype=np.int32)
            a = freqs_minus_T[np.argmin(freqs_minus_T.sum(axis=1))]
            for b in T:
                query = a*b
                key = tuple(query.tolist())

                if key in self.S:
                    continue 
                
                if key in self.measurements:
                    measurement = self.measurements[key]
                else: 
                    measurement = s(query)[0]
                    self.n_queries += 1
                    self.measurements[key] = measurement
                
                if self.print_flag:
                    print('trying', query)
                    print(measurement, s_est(query)[0])
                
                if np.abs(measurement - s_est(query)[0]) > self.tres:
                    if self.print_flag:
                        print('\t found counter example', query)
                        #raise Exception('found a counter example')
                    return query
                else:
                    if key not in self.S:
                        self.Snew[key] = measurement
                    self.S[key] = measurement  
                    
                    freqs = np.concatenate([freqs, query[np.newaxis]], axis=0) 
                    
                    
            T += [a]
        return None
    
    def addElement(self, s, c):
        if self.print_flag:
            print('adding', c)
        if len(self.S) == 0:
            ckey = tuple(c.tolist())
            measurement = s(c)[0]
            self.n_queries += 1
            self.measurements[ckey] = measurement
            if ckey not in self.S:
                self.Snew[ckey] = measurement
            self.S[ckey] = self.measurements[ckey]
        else:
            l = c.sum()
            s_est = self.hyp()
            freqs = s_est.freqs
            while l > 0:
                for a in freqs:
                    if a.sum() == l:
                        query = a * c
                        key = tuple(query.tolist())
                        if key in self.measurements:
                            measurement = self.measurements[key]
                            continue
                        else:
                            measurement = s(query)[0]
                            self.n_queries += 1
                            self.measurements[key] = measurement
                            if np.abs(measurement - s_est(query)[0]) > self.tres:
                                c = query
                                break
                l = min(l-1, c.sum())
            # update S
            if self.print_flag:
                print('added', c)
            ckey = tuple(c.tolist())
            if ckey not in self.S:
                self.S[ckey] = self.measurements[ckey]
                self.Snew[ckey] = self.measurements[ckey]
            for a in freqs:
                if a.sum() < c.sum():
                    query = c * a
                    key = tuple(query.tolist())
                    if key not in self.measurements:
                        measurement = s(query)[0]
                        self.n_queries += 1
                        self.measurements[key] = measurement
                    if key not in self.S:
                        self.S[key] = self.measurements[key]
                        self.Snew[key] = self.measurements[key]
            if self.print_flag:
                print('added', len(self.Snew), 'keys')
                
    def equiv(self, s):
        if len(self.S) == 0:
            query = self.equiv_queries[0]
            self.measurements[tuple(query.tolist())] = s(query)[0]
            self.n_queries += 1
            return query
        
        s_est = self.hyp()
        
        queries = self.equiv_queries
        perm = np.random.permutation(len(queries))
        for c in queries[perm]:
            key = tuple(c.tolist())
            if key in self.measurements:
                val = self.measurements[key]
            else:
                val = s(c)[0]
                self.n_queries += 1
                self.measurements[key] = val
            if np.abs(s_est(c)[0] - val) > self.tres:
                return c
        return None
        
    def computeError(self, metric = lambda vec_true, vec_est: np.linalg.norm(vec_true - vec_est)/np.linalg.norm(vec_true)):
        s_est = self.hyp()
        return metric(self.error_measurements, s_est(self.error_queries))
    
    def getStats(self):
        if self.record_error:
            return self.stats
        else:
            return None
    
    def transform(self, s):
        try:
            self.measurements = {}
            self.S = {}
            self.Snew = {}
            if self.equivalence_budget == 2**self.n:
                queries = np.asarray([int2indicator(A, self.n) for A in range(2**self.n)], dtype=np.int32)
            else:
                queries = np.random.binomial(1, 0.5, (self.equivalence_budget, self.n)).astype(np.int32)
            self.equiv_queries = queries
            
            if self.record_error:
                if self.equivalence_budget == 2**self.n:
                    queries = np.asarray([int2indicator(A, self.n) for A in range(2**self.n)], dtype=np.int32)
                else:
                    queries = np.random.binomial(1, 0.5, (self.equivalence_budget, self.n)).astype(np.int32)
                self.error_queries = queries
                self.error_measurements = s(queries, count_flag=False)
                self.stats = defaultdict(list)
            
            
            #values = s(queries)
            #self.equiv_queries = {tuple(q.tolist()): v for q, v in zip(queries, values)}
            #self.equiv_queries_tuple = (queries, values)
            print()
            with tqdm.tqdm(total=self.total_budget, file=sys.stdout) as pbar:
                while len(self.measurements) < self.total_budget:
                    len_before = len(self.measurements) 
                    if self.print_flag:
                        print('equivalence query...')
                    c = self.equiv(s)
                    if c is None:
                        break
                    while True:
                        if self.print_flag:
                            print('adding element...')
                        self.addElement(s, c)
                        if self.record_error:
                            self.stats['n_queries_used_to_fit'] += [len(self.S)]
                            self.stats['n_queries_total'] += [len(self.measurements)]
                            self.stats['n_nonzeros'] += [len(self.hyp().coefs)]
                            self.stats['errors'] += [self.computeError()]
                            
                        if self.print_flag:
                            print('searching for counter example...')
                        c = self.easyCounterExample(s)
                        if c is None:
                            if self.print_flag:
                                print('\t no counter example found')
                            break
                    len_after = len(self.measurements)
                    pbar.update(len_after - len_before)
        except KeyboardInterrupt:
            if self.print_flag:
                print('Early stopping via keyboardinterrupt.')
        finally:
            if self.print_flag:
                print('finally part...')
            if self.flag_refit:
                self.S = self.measurements
            return self.hyp()
            
            
                

