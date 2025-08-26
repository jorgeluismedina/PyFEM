
import numpy as np

class Material:
    def __init__(self, elast, poiss, dense):
        self.elast = elast
        self.poiss = poiss
        self.dense = dense

class Metal(Material):
    def __init__(self, elast, poiss, dense, hards, yield_stress):
        super().__init__(elast, poiss, dense)
        self.hards = hards
        self.yield_stress = yield_stress

class CSR(Material):
    def __init__(self, elast, poiss, dense, hards, frict, cohes):
        super().__init__(elast, poiss, dense)
        self.hards = hards
        self.frict = frict
        self.cohes = cohes