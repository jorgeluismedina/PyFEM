
import numpy as np
from pyfem.materials.const_models import get_constitutive_matrix 

class Material:
    def __init__(self, elast, poiss, dense, constitutive_model):
        self.elast = elast
        self.poiss = poiss
        self.dense = dense
        self.dmatx = get_constitutive_matrix(elast, poiss, constitutive_model)

class Metal(Material):
    def __init__(self, elast, poiss, dense, hards, sigma_yield, constitutive_law):
        super().__init__(elast, poiss, dense, constitutive_law)
        self.hards = hards # hardening parameter
        self.sigma_yield = sigma_yield

class CSR(Material):
    def __init__(self, elast, poiss, dense, hards, frict, cohes, constitutive_law):
        super().__init__(elast, poiss, dense, constitutive_law)
        self.hards = hards # hardening parameter
        self.frict = frict
        self.cohes = cohes