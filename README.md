# Learning Set Functions that are Sparse in Better 
# Non-Orthogonal Fourier Bases


This repository, accompaniyng the Bachelor thesis "Learning Set Functions that are Sparse in Better Non-Orthogonal Fourier Bases", provides implementations for different sparse set function transforms (SSFT) and for various set functions.

## Experiments and installation

First, note that the code in this repository is an extension of the work of the authors of \[1\]. Hence code that is partly identical to this one can be found in the [repository](https://github.com/chrislybaer/aaai-ssft) accompanying \[1\].

We ran everything with Python 3.8.

The auction simulation experiments require pyjnius, for its installation we refer to the [repository](https://github.com/chrislybaer/aaai-ssft) accompanying \[1\].
When you download sats-v0.6.4.jar, place it in ./aaai-ssft-master/exp/datasets/PySats/lib.

For running the compiler flag optimization tests install [compiler gym](https://compilergym.com/index.html)
```bash
pip install -U compiler_gym
```
as well as [CK](https://ck.readthedocs.io/en/latest/index.html)
```bash
pip install ck
```
Additionally install the [benchmark datasets](https://github.com/ctuning/ctuning-programs) via the CK framework
```bash
ck pull repo:ck-autotuning
ck pull repo:ck-env

ck pull repo:ctuning-programs
ck pull repo:ctuning-datasets-min
```

For other installation methods or in case of issues with the installation of compiler gym or CK, please consult
* https://compilergym.com/getting_started.html 
* https://ck.readthedocs.io/en/latest/src/installation.html#ck-installation

Other dependencies include
* [numpy](https://numpy.org/)
* [scipy](https://scipy.org/)
* [matplotlib](https://matplotlib.org/)
* [sympy](https://www.sympy.org/en/index.html) for performing the WHT
* [sklearn](https://scikit-learn.org/stable/) for training the random forest regressors
* [sacred](https://sacred.readthedocs.io/en/stable/index.html) for performing the experiments

You can install these remaining requirements, and others, with pip

```bash
pip install -r requirements.txt
```

## SSFT algorithms

We present implementations for three Fourier transforms for sparse set functions. Two of which were already implemented by the authors of [1] and can also be found [here](https://github.com/chrislybaer/aaai-ssft). Here we introduce a first implementation of the SSFTW3 algorithm.


## Set functions 

We also present implementations for six classes of set functions. Three of which (Sensor placement tasks, preference functions and preference elicitation in auctions) were also already introduced and implemented in [1] and can be found in the corresponindg [repository](https://github.com/chrislybaer/aaai-ssft). The remaining three are newly implemented by us: fitness functions, random forest regressors on binary input data and compiler flag optimization tasks. 

To perform the experiments we used sacred. The -F flag specifies the directory to store logs and results. For each run there will be a folder in the specified directory, containing the files 
* cout.txt with the standard output information of the run 
* run.json with the resulted support and coefficients
* metrics.json with some intermediate results (relative error, number of queries, number of recovered coefficients etc.)
* config.json with the input parameters


### Fitness Functions 

To run the fitness function experiments run:

```bash
python -m exp.run_fitness with model.SSFTW3 dataset.BANK -F target_dir 
```

### Random forest regressors

To run the random forest regressor task run:

```bash
python -m exp.run_rfr with model.SSFT3 dataset.SUPERCOND -F target_dir 
```

### Compiler flag optimizations

To run the object file size task run

```bash
python -m exp.run_objsize with model.SSFT4 dataset.OBJSIZE -F target_dir 
```
and for the execution time task

```bash
python -m exp.run_exectime with model.SSFT4 dataset.SUSAN -F target_dir 
```

## Plots

Additionaly, if you wish to plot the Fourier spectrums of a set function, run

```py
import matmulFT
from exp.ingredients import compiler_dataset as dataset

set_function, n = dataset.load_objsize(n=5)
matmulFT.plot(set_function, n)

```
from a Python interpreter in the 'aaai-ssft-master' folder.
Note that matmultFT performs the transforms via matrix multiplication. 


## References
\[1\]: 
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