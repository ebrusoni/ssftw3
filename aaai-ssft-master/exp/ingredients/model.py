from copyreg import constructor
import inspect

from sacred import Ingredient
from sdsft import SparseSFT
from sdsft.transforms.learnpoly import LearnPolyDSFT4

ingredient = Ingredient('model')
K_MAX = 1000
EPS = 1e-2

@ingredient.config
def cfg():
    """Model configuration."""
    name = ''
    constructor = None
    parameters = {}


@ingredient.named_config
def SSFTW3():
    name = 'Sparse set function Fourier transform'
    constructor = SparseSFT
    parameters = {
        'eps': 1e-3,
        'flag_print':False,
        'k_max': K_MAX,
        'flag_general':False,
        'model':'W3'
    }

@ingredient.named_config
def SSFT4():
    name = 'Sparse set function Fourier transform'
    constructor = SparseSFT
    parameters = {
        'eps':1e-8,
        'flag_print':False,
        'k_max':K_MAX,
        'flag_general':False,
        'model':'4'
    }

@ingredient.named_config
def LEARNPOLY4():
    name = 'Sparse set function Fourier transform'
    constructor = LearnPolyDSFT4
    parameters = {
        'tres':1e-6,
        'flag_print':False,
        'equivalence_budget':10000,
        'total_budget':100000,
        'record_error':True,
        'flag_refit':False
    }


@ingredient.named_config
def SSFT3():
    name = 'Sparse set function Fourier transform'
    constructor = SparseSFT
    parameters = {
        'eps':1e-3,
        'flag_print':False,
        'k_max': K_MAX,
        'flag_general':False,
        'model':'3'
    }

@ingredient.named_config
def SSFT4Plus():
    name = 'Sparse set function Fourier transform plus filtering'
    constructor = SparseSFT
    parameters = {
        'eps':1e-3,
        'flag_print':False,
        'k_max':1000,
        'flag_general':True,
        'model':'4'
    }

@ingredient.named_config
def SSFT3Plus():
    name = 'Sparse set function Fourier transform plus filtering'
    constructor = SparseSFT
    parameters = {
        'eps':1e-3,
        'flag_print':False,
        'k_max':1000,
        'flag_general':True,
        'model':'3'
    }



@ingredient.capture
def get_instance(n, name, constructor, parameters, _log):
    """Get an instance of a model according to parameters in the configuration.

    Also, check if the provided parameters fit to the signature of the model
    class and log default values if not defined via the configuration.

    """
    signature = inspect.signature(constructor)
    available_parameters = signature.parameters
    valid_parameters = {}
    #remove unnecessary parameters
    for key in parameters.keys():
        if key in available_parameters.keys():
            valid_parameters[key] = parameters[key]

    return constructor(n, **valid_parameters)