
import numpy as np
from abc import ABC, abstractmethod
from .yield_criterion import get_yield_criterion
from .const_models import get_const_model

class Material(ABC):
    def __init__(self, elast, poiss, hards, dense):
        self.elast = elast
        self.poiss = poiss
        self.hards = hards
        self.dense = dense
        self.yield_crite = None 
        self.const_model = None

    def add_constitutive_model(self, problem_type):
        self.const_model = get_const_model(problem_type)(self)   
    
    def calculate_dmatx(self, npoin):
        D = self.const_model.calculate_D()
        return np.tile(D,(npoin,1,1))

    def calculate_Dp(self, avect):
        return self.const_model.calculate_Dp(avect)
    
    @abstractmethod
    def add_yield_criterion(self, yield_criterion):
        pass


class Metal(Material):
    def __init__(self, elast, poiss, hards, uniax, dense):
        super().__init__(elast, poiss, hards, dense)
        self.uniax = uniax

    def add_yield_criterion(self, yield_criterion):
        self.yield_crite = get_yield_criterion(yield_criterion)(self)

class CSR(Material):
    def __init__(self, elast, poiss, hards, frict, cohes, dense):
        super().__init__(elast, poiss, hards, dense)
        self.frict = frict
        self.cohes = cohes

    def add_yield_criterion(self, yield_criterion):
        self.yield_crite = get_yield_criterion(yield_criterion)(self)