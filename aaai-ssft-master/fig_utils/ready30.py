from unittest import result
import numpy as np
import pandas as pd

import sdsft.transforms.canonical as can
import sdsft.common as com

def test(s, n, model, eps):
    flag_general = False
    if model == '4':
        flag_general = False
    SSFT3 = can.SparseSFT(n, flag_general=flag_general, flag_print=False, model=model, eps=eps, k_max=1000)
    wrapped_s = com.WrapSetFunction(s, use_loop=True)
    estimate = SSFT3.transform(wrapped_s)
    #print(estimate.freqs)
    #print(estimate.coefs)
    err = com.eval_sf(wrapped_s, estimate, n, n_samples=1000, err_type='rel')
    print(f'model: {model} -> error:{err}, no. of coefs: {len(estimate.coefs)}')
    return (err, len(estimate.coefs))


# N=13
# results = []
# for i in range(1):
#     K = 7
#     f = NK(N,K)
#     print(f'N:{N}, K:{K}')
#     one_run = []
#     for model in ['3', '4', 'W3']: #,'3', 'W3'
#         #print(f'model: {model}')
#         err, coeff= test(f.eval, N, model, 10e-8)
#         one_run.append([err, coeff])
#         #print('\n')
#     results.append(one_run)
# r = np.array(results)
#print(f'variance: \n{np.var(r, axis=0)} \n mean: \n{np.mean(r, axis=0)}')
#print(f'evaluation at index1 {index1}: {f.eval(index1)}\nand index2 {index2}: {f.eval(index2)}\nneighbours: {f.neighs} \nneighbour function: {f.neighs_eval}')

#Paper https://www.nature.com/articles/s41467-019-12130-8#Sec20
df = pd.read_excel('C:/Users/henry/OneDrive/Desktop/BachelorThesis/CSVdatasets/n13.xlsx', engine='openpyxl')
df = df[1:]
s = FitnessFunction(df, (df.columns[0], df.columns[7]), 13, fitcom.get_idx_RED_BLUE)
test(s.eval2, 13, '3', 10e-8)
test(s.eval2, 13, '4', 10e-8)
test(s.eval2, 13, 'W3', 10e-8)