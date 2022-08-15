# Learning Set Functions that are Sparse in Better Non-Orthogonal Fourier Bases


This repository underlying the Bachelor thesis "Learning Set Functions that are Sparse in Better Non-Orthogonal Fourier Bases" provides implementations for different sparse set function transforms (SSFT) and for various classes of set functions.

## Experiments and installation

For details on sacred, the python package we used to perform the exepriments, and the correct installation of pyjnius, we refer to the repository accompanying [1](*1).
After having installed pyjnius you can simply install the remainig packages in the requirements file:
```bash
pip install -r requirements.txt
```
For running the compiler flag execution time tasks you have to install the compiler benchmarks (*2) using CK (Collective Knowledge):

```bash
ck pull repo:ctuning-programs

ck pull repo:ctuning-datasets-min
```


*1: https://github.com/chrislybaer/aaai-ssft
*2: https://github.com/ctuning/ctuning-programs
## SSFT algorithms

We present implementations for various three Fourier transforms for sparse set functions. Two of those were already implemented by the authors of [1]. Here we introduce a first implementation of the SSFTW3 algorithm.


## Set functions 

We also present implementations for six classes of set functions. Three of which (Sensor placement tasks, preference functions and prefernece elicitation in auctions) we were already introduced and implemented in [1]. The remaining three are newly implemented by us: fitness functions, random forest regressors on binary input data and compiler flag optimization tasks. 

### Fitness Functions 

To run the fitness function experiments run:

```bash
python -m exp.run_fitness with model.SSFTW3 dataset.BANK -F target_dir 
```

### Random forest regressors

To run the random forest regressor task run:

```bash
python -m exp.run_decisiontree with model.SSFT3 dataset.SUPERCOND -F target_dir 
```

### Compiler flags

To run the object file size task run

```bash
python -m exp.run_objsize with model.SSFT4 dataset.OBJSIZE -F target_dir 
```
and for the execution time task

```bash
python -m exp.run_exectime with model.SSFT4 dataset.SUSAN -F target_dir 
```

## References
[1]: 
```bibtex
@article{Wendler_Amrollahi_Seifert_Krause_Püschel_2021, 
title={Learning Set Functions that are Sparse in Non-Orthogonal {F}ourier Bases}, 
volume={35}, 
url={https://ojs.aaai.org/index.php/AAAI/article/view/17232}, 
number={12}, 
journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
author={Wendler, Chris and Amrollahi, Andisheh and Seifert, Bastian and Krause, Andreas and P{\"u}schel, Markus}, 
year={2021}, 
month={May}, 
pages={10283-10292}
}
```