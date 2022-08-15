import inspect
import pandas as pd

from sacred import Ingredient

import functools
from ..fitness_utils import fitness_common as fcom
from ..fitness_utils.FitnessFunction import FitnessFunction as FitF
from ..fitness_utils.NK_model import NK_model as NK
import sdsft
import numpy as np
from sdsft.common import WrapSetFunction

ingredient = Ingredient('dataset')

@ingredient.config
def cfg():
    """Dataset configuration"""
    name = ''
    set_function = None
    n = None

def load_red():
    #Paper https://www.nature.com/articles/s41467-019-12130-8#Sec20
    df = pd.read_excel('./exp/datasets/fitness/Poelwijk2019.xlsx', engine='openpyxl')
    df = df[1:]
    s = FitF(df, (df.columns[0], df.columns[6]), 13, fcom.get_idx_RED_BLUE)
    wrapped_s = WrapSetFunction(s, use_loop=True)
    return (wrapped_s, 13)

def load_blue():
    #Paper https://www.nature.com/articles/s41467-019-12130-8#Sec20
    df = pd.read_excel('./exp/datasets/fitness/Poelwijk2019.xlsx', engine='openpyxl')
    df = df[1:]
    s = FitF(df, (df.columns[0], df.columns[7]), 13, fcom.get_idx_RED_BLUE)
    wrapped_s = WrapSetFunction(s, use_loop=True)
    return (wrapped_s, 13)

def load_rtsgp():
    df = pd.read_csv('./exp/datasets/fitness/genotype_fitness.txt', delimiter='\t')
    df = df.groupby(by='genotype', as_index=False).mean()
    s = FitF(df, (df.columns[0], df.columns[2]), 5, fcom.get_idx_RTSGP)
    wrapped_s = WrapSetFunction(s, use_loop=True)
    return (wrapped_s,  5)

def load_weinreich():
    #Paper: https://www.science.org/doi/10.1126/science.1123539 (sign epistasis is prevalent)
    df = pd.read_csv('./exp/datasets/fitness/Weinreich2006.csv', sep=';')
    Ind = fcom.merge_idx_cols(df, 0, -1)
    IndVal = pd.concat([Ind, np.log(df.iloc[:,-1])], axis=1)
    s = FitF(IndVal, (IndVal.columns[0], IndVal.columns[1]), 5, fcom.get_idx_FIRST)
    wrapped_s = WrapSetFunction(s, use_loop=True)
    return (wrapped_s, 5)

def load_geno_pheno():
    df = pd.read_csv('./exp/datasets/fitness/cas9_binary_1nm_full.csv')
    Ind = fcom.merge_idx_cols(df, 1, 24)
    IndVal = pd.concat([Ind, df.iloc[:,-1]], axis=1)
    s = FitF(IndVal, IndVal.columns, 23, fcom.get_idx_genopheno)
    wrapped_s = WrapSetFunction(s, use_loop=True)
    return(wrapped_s, 23)

def load_franke():
    df = pd.read_csv('./exp/datasets/fitness/Franke2011.csv')
    Ind = fcom.merge_idx_cols(df, 1, 9)
    IndVal = pd.concat([Ind, df.iloc[:,-1]], axis=1)
    s = FitF(IndVal, IndVal.columns, 8, fcom.get_idx_genopheno)
    wrapped_s = WrapSetFunction(s, use_loop=True)
    return(wrapped_s, 8)

def load_bank():
    #github: https://github.com/weinreichlab/JStatPhys2018/tree/master/Datasets
    df = pd.read_csv('./exp/datasets/fitness/Bank2016.csv')
    Ind = fcom.merge_idx_cols(df, 0, 6)
    IndVal = pd.concat([Ind, df.iloc[:,-2]], axis=1)
    s = FitF(IndVal, IndVal.columns, 6, fcom.get_idx_common)
    wrapped_s = WrapSetFunction(s, use_loop=True)
    return (wrapped_s, 6)

def load_palmer():
    #github: https://github.com/weinreichlab/JStatPhys2018/tree/master/Datasets
    df = pd.read_csv('./exp/datasets/fitness/Palmer2015.csv')
    Ind = fcom.merge_idx_cols(df, 0, 6)
    IndVal = pd.concat([Ind, df.iloc[:,-2]], axis=1)
    s = FitF(IndVal, IndVal.columns, 6, fcom.get_idx_common)
    wrapped_s = WrapSetFunction(s, use_loop=True)
    return (wrapped_s, 6)

def load_hall():
    #dataset from: https://github.com/weinreichlab/JStatPhys2018/tree/master/Datasets
    #paper: https://doi.org/10.1093/jhered/esq007
    df = pd.read_csv('./exp/datasets/fitness/Hall2010.csv')
    Ind = fcom.merge_idx_cols(df, 0, 6)
    IndVal = pd.concat([Ind, df.iloc[:,-4]], axis=1)
    s = FitF(IndVal, IndVal.columns, 6, fcom.get_idx_common)
    wrapped_s = WrapSetFunction(s, use_loop=True)
    return (wrapped_s, 6)

def load_omaille(no=2):
    #dataset from: https://github.com/weinreichlab/JStatPhys2018/tree/master/Datasets
    #paper: https://doi:10.1038/nchembio.113
    df = pd.read_csv('./exp/datasets/fitness/OMaille2008.csv')
    Ind = fcom.merge_idx_cols(df, 0, 6)
    IndVal = pd.concat([Ind, df.iloc[:,-4+no]], axis=1)
    s = FitF(IndVal, IndVal.columns, 6, fcom.get_idx_common)
    s = fcom.permute_fs(s, (4, 2, 0, 3, 5, 1))
    wrapped_s = WrapSetFunction(s, use_loop=True)
    return (wrapped_s, 6)

def load_khan():
    #dataset from: https://github.com/weinreichlab/JStatPhys2018/tree/master/Datasets
    #paper: https://doi:10.1126/science.1203801 for growth rates (Table S2)
    df = pd.read_csv('./exp/datasets/fitness/Khan2011.csv')
    Ind = fcom.merge_idx_cols(df, 0, 5)
    IndVal = pd.concat([Ind, df.iloc[:,-4]], axis=1)
    s = FitF(IndVal, IndVal.columns, 5, fcom.get_idx_common)
    wrapped_s = WrapSetFunction(s, use_loop=True)
    return (wrapped_s, 5)

def load_lunzer():
    #dataset from: https://github.com/weinreichlab/JStatPhys2018/tree/master/Datasets
    df = pd.read_csv('./exp/datasets/fitness/Lunzer2005.csv')
    Ind = fcom.merge_idx_cols(df, 0, 6)
    IndVal = pd.concat([Ind, df.iloc[:,-1]], axis=1)
    s = FitF(IndVal, IndVal.columns, 6, fcom.get_idx_common)
    wrapped_s = WrapSetFunction(s, use_loop=True)
    return (wrapped_s, 6)

def load_whitlock():
    #dataset from: https://github.com/weinreichlab/JStatPhys2018/tree/master/Datasets
    #doi: http://www.jstor.org/stable/2640663
    df = pd.read_csv('./exp/datasets/fitness/Whitlock2000.csv')
    Ind = fcom.merge_idx_cols(df, 0, 5)
    IndVal = pd.concat([Ind, df.iloc[:,-1]], axis=1)
    s = FitF(IndVal, IndVal.columns, 5, fcom.get_idx_common)
    wrapped_s = WrapSetFunction(s, use_loop=True)
    return (wrapped_s, 5)

def load_dasilva():
    #dataset from: https://github.com/weinreichlab/JStatPhys2018/tree/master/Datasets
    #doi: https://doi.org/10.1534/genetics.109.112458
    df = pd.read_csv('./exp/datasets/fitness/DaSilva2010.csv')
    Ind = fcom.merge_idx_cols(df, 0, 5)
    IndVal = pd.concat([Ind, df.iloc[:,-2]], axis=1)
    s = FitF(IndVal, IndVal.columns, 5, fcom.get_idx_common)
    wrapped_s = WrapSetFunction(s, use_loop=True)
    return (wrapped_s, 5)

def load_devisser():
    #dataset from: https://github.com/weinreichlab/JStatPhys2018/tree/master/Datasets
    #doi: https://doi.org/10.1534/genetics.109.112458
    df = pd.read_csv('./exp/datasets/fitness/DeVisser2009.csv')
    Ind = fcom.merge_idx_cols(df, 0, 5)
    IndVal = pd.concat([Ind, df.iloc[:,-2]], axis=1)
    s = FitF(IndVal, IndVal.columns, 5, fcom.get_idx_common)
    wrapped_s = WrapSetFunction(s, use_loop=True)
    return (wrapped_s, 5)


def load_nk(n, k):
    # n = 10
    # k = 1
    s = NK(n,k)
    wrapped_s = WrapSetFunction(s, use_call_dict=True)
    return (wrapped_s, n)




@ingredient.named_config
def RED():
    name = 'entacmaea_quadricolor_red'
    set_function, n = load_red()

@ingredient.named_config
def BLUE():
    name = 'entacmaea_quadricolor_blue'
    set_function, n = load_blue()

@ingredient.named_config
def RTSGP():
    name = 'rtsgp'
    set_function, n = load_rtsgp()

@ingredient.named_config
def WEINREICH():
    name = 'weinreich2006'
    set_function, n = load_weinreich()

@ingredient.named_config
def GENOPHENO():
    name = 'genopheno'
    set_function, n = load_geno_pheno()

@ingredient.named_config
def FRANKE():
    name = 'franke2011'
    set_function, n = load_franke()

@ingredient.named_config
def BANK():
    name = 'Bank2016'
    set_function, n = load_bank()

@ingredient.named_config
def PALMER():
    name = 'Palmer2015'
    set_function, n = load_palmer()

@ingredient.named_config
def HALL():
    name = 'Hall2010'
    set_function, n = load_hall()

@ingredient.named_config
def OMAILLE():
    name = 'OMaille 2008'
    set_function, n = load_omaille()

@ingredient.named_config
def KHAN():
    name = 'Khan 2011'
    set_function, n = load_khan()

@ingredient.named_config
def LUNZER():
    name = 'Lunzer 2005'
    set_function, n = load_lunzer()

@ingredient.named_config
def WHITLOCK():
    name = 'Whitlock 2000'
    set_function, n = load_whitlock()

@ingredient.named_config
def DASILVA():
    name = 'DaSilva 2010'
    set_function, n = load_dasilva()

@ingredient.named_config
def DEVISSER():
    name = 'DeVisser 2009'
    set_function, n = load_devisser()

@ingredient.named_config
def NKMODEL():
    name = 'NK model'
    set_function, n = load_nk(10, 4)

@ingredient.capture
def get_instance(name, n, set_function, _log):
    """Get an instance of a model according to parameters in the configuration.

    Also, check if the provided parameters fit to the signature of the model
    class and log default values if not defined via the configuration.

    """
    s = set_function
    return s, n