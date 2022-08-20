from copyreg import constructor
import inspect

from sacred import Ingredient
from sdsft import SparseSFT

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
        'eps':1e-3,
        'flag_print':False,
        'k_max':K_MAX,
        'flag_general':False,
        'model':'4'
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
        'eps':1e-10,
        'flag_print':True,
        'k_max':1000,
        'flag_general':True,
        'model':'4'
    }

@ingredient.named_config
def SSFT4la():
    name = 'Sparse set function Fourier transform plus filtering'
    constructor = SparseSFT
    parameters = {
        'eps':1e-8,
        'flag_print':True,
        'k_max':1000,
        'flag_general':False,
        'naive_fix':True,
        'model':'4'
    }

@ingredient.named_config
def SSFT3la():
    name = 'Sparse set function Fourier transform plus filtering'
    constructor = SparseSFT
    parameters = {
        'eps':1e-10,
        'flag_print':True,
        'k_max':1000,
        'flag_general':False,
        'naive_fix':True,
        'model':'3'
    }

@ingredient.named_config
def SSFT3Plus():
    name = 'Sparse set function Fourier transform plus filtering'
    constructor = SparseSFT
    parameters = {
        'eps':1e-3,
        'flag_print':True,
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
