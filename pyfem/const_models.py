
import numpy as np
from abc import ABC, abstractmethod
from .stress_utils import get_yield_criterion

class ElastoPlastic2D(ABC):
    def __init__(self, material):
        self.mater = material

    @abstractmethod
    def calculate_D(self):
        pass

    @abstractmethod
    def get_fmult1(self, avect):
        pass

    @abstractmethod
    def prepare_stress(self, stress):
        pass

    def get_flowpl(self, avect):
        elast = self.mater.elast
        poiss = self.mater.poiss
        fmul1 = elast/(1+poiss)
        fmult = self.get_fmult1(avect)
        dvect = np.array([fmul1*avect[0] + fmult, 
                          fmul1*avect[1] + fmult,
                          fmul1*0.5*avect[2],
                          fmul1*avect[3] + fmult])
        return dvect
    
    def calculate_Dp(self, avect):
        D = self.calculate_D()
        dimen = D.shape[0]
        dvect = self.get_flowpl(avect)
        denom = self.mater.hards + dvect@avect
        dDdDT = np.tensordot(dvect,dvect, axes=0)[:dimen, :dimen]
        return D - dDdDT/denom
        

        
    
class PlaneStress(ElastoPlastic2D):
    def calculate_D(self):
        elast = self.mater.elast
        poiss = self.mater.poiss
        const = elast/(1-poiss**2)
        conss = (1-poiss)/2
        D = const*np.array([[1, poiss, 0],
                            [poiss, 1, 0],
                            [0, 0, conss]])
        return D
    
    def get_fmult1(self, avect):
        elast = self.mater.elast
        poiss = self.mater.poiss
        avect_sum = avect[0]+avect[1]
        return elast*poiss*(avect_sum)/(1-poiss**2)
    
    def prepare_stress(self, stress):
        mod_stress = np.insert(stress, 3, 0, axis=1)
        return mod_stress
        

class PlaneStrain(ElastoPlastic2D):
    def calculate_D(self):
        elast = self.mater.elast
        poiss = self.mater.poiss
        const = elast*(1-poiss) / ((1+poiss)*(1-2*poiss))
        conss = poiss/(1-poiss)
        cons2 = (1-2*poiss) / (2*(1-poiss))
        D = const*np.array([[1, conss, 0],
                            [conss, 1, 0],
                            [0, 0, cons2]])
        return D
    
    def get_fmult1(self, avect):
        elast = self.mater.elast
        poiss = self.mater.poiss
        avect_sum = avect[0]+avect[1]+avect[3]
        return elast*poiss*(avect_sum)/((1+poiss)*(1-2*poiss))
    

class Axisymmetric(ElastoPlastic2D):
    def calculate_D(self):
        elast = self.mater.elast
        poiss = self.mater.poiss
        const = elast*(1-poiss) / ((1+poiss)*(1-2*poiss))
        conss = poiss/(1-poiss)
        cons2 = (1-2*poiss) / (2*(1-poiss))
        D = const*np.array([[1, conss, 0, conss],
                            [conss, 1, 0, conss],
                            [0, 0, cons2, 0],
                            [conss, conss, 0, 1]])
        return D
    
    def get_fmult1(self, avect):
        elast = self.mater.elast
        poiss = self.mater.poiss
        avect_sum = avect[0]+avect[1]+avect[3]
        return elast*poiss*(avect_sum)/((1+poiss)*(1-2*poiss))
    



def get_const_model(problem_type):
    problem_types = {'PlaneStress': PlaneStress, 
                     'PlaneStrain': PlaneStrain,
                     'Axisymmetric': Axisymmetric}
    
    return problem_types.get(problem_type)