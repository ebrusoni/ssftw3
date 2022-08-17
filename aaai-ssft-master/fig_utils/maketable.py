import numpy as np
import re

'''
Helper file for generating tables for preference elicitation in auctions.

Usage:

1. Run elicitation experiment from command line, for example :  python -m exp.run_elicitation with model.SSFT4 dataset.MRVM -F target_dir

2. Sacred will output a folder in <target_dir> containing the details of the run. Assign the path string of the <cout.txt> to the <cout> variable below 

'''
def get_stuff(bidders):
    means = np.mean(bidders, axis=0)
    maxs = np.max(bidders, axis=0)
    mins = np.min(bidders, axis=0)

    # diff_max = maxs-means
    # diff_min = means-mins
    # print(f'rel: {means[0]} +- {max(diff_max[0], diff_min[0])}')
    # print(f'q: {means[1]} +- {max(diff_max[1], diff_min[1])}')
    # print(f'k: {means[2]} +- {max(diff_max[2], diff_min[2])}')

    diff_max = maxs-means
    diff_min = means-mins
    print(bidders[:,0].shape)
    print(f'rel: {means[0]} +- {np.std(bidders[:, 0])}')
    print(f'q: {means[1]} +- {np.std(bidders[:, 1])}')
    print(f'k: {means[2]} +- {np.std(bidders[:, 2])}')
    print()

cout = '/home/enri/code/ba/aaai-ssft-master/results/elicitation/MRVM/7/cout.txt'
with open(cout) as f:
    lines = f.readlines()
lines = [re.findall("\d*\.?\d+", s) for s in lines if len(s.split(','))==7]#.split(',')
rel = 4
q = 5
k = 7

print(lines)

lines = np.array(lines, dtype=np.float32)[:, [rel, q, k]]

local = np.zeros(shape=(30,3))
regional = np.zeros(shape=(40,3))
national = np.zeros(shape=(30,3))

for i in range(0,10,1):
    local[3*i:3*(i+1)] = lines[10* i: 10* i+3]
    regional[4*i:4*(i+1)]= lines[10*i+3:10*i+7]
    national[3*i:3*(i+1)]= lines[10*i+7:10*i+10]

print(local)
get_stuff(local)
get_stuff(regional)
get_stuff(national)