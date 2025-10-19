
import numpy as np
from abc import ABC, abstractmethod



def plane_stress(elast, poiss):
    const = elast / (1 - poiss**2)
    D = np.array([
        [1,  poiss,  0],
        [poiss,  1,   0],
        [0,  0,   (1 - poiss)/2]
    ]) * const
    return D

def plane_strain(elast, poiss):
    const = elast / ((1 + poiss) * (1 - 2 * poiss))
    D = np.array([
        [1 - poiss,  poiss,  0],
        [poiss,  1 - poiss,   0],
        [0,  0,   (1 - 2*poiss)/2]
    ]) * const
    return D

def axisymmetric(elast, poiss):
    const = elast / ((1 + poiss) * (1 - 2 * poiss))
    D = const * np.array([
        [1 - poiss,  poiss,      poiss,     0],
        [poiss,      1 - poiss,  poiss,     0],
        [poiss,      poiss,      1 - poiss, 0],
        [0,      0,      0,      (1 - 2*poiss)/2]
    ]) * const
    return D


def elastic3D(elast, poiss):
    lame1 = elast * poiss / (1 + poiss) / (1 - 2*poiss) #lambda
    lame2 = 0.5 * elast / (1 + poiss) #mu
    const = 2*lame2 + lame1
    D = np.array([
        [const, lame1, lame1, 0, 0, 0],
        [lame1, const, lame1, 0, 0, 0],
        [lame1, lame1, const, 0, 0, 0],
        [0, 0, 0, lame2, 0, 0],
        [0, 0, 0, 0, lame2, 0],
        [0, 0, 0, 0, 0, lame2]
    ])
    return D




_constitutive_model_ = {'plane_stress': plane_stress, 
                        'plane_strain': plane_strain,
                        'axisymmetric': axisymmetric,
                        'elastic3D': elastic3D}

def get_constitutive_matrix(elast, poiss, name):
    if name not in _constitutive_model_:
        raise ValueError(f"Constitutive law no encountered")
    
    return _constitutive_model_[name](elast, poiss)







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
    def all_components(self, stress):
        pass

    def get_flowpl(self, elast, poiss, avect):
        fmul1 = elast/(1+poiss)
        fmult = self.get_fmult1(avect)
        dvect = np.array([fmul1*avect[0] + fmult, 
                          fmul1*avect[1] + fmult,
                          fmul1*0.5*avect[2],
                          fmul1*avect[3] + fmult])
        return dvect
    
    
        



        
#'''    
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
    
    #def all_components(self, stress):
    #    mod_stress = np.insert(stress, 3, 0, axis=1)
    #    return mod_stress
    def calculate_Dp(self, elast, poiss, avect):
        D = self.calculate_D()
        dimen = D.shape[0]
        dvect = self.get_flowpl(elast, poiss, avect)
        denom = self.mater.hards + dvect@avect
        dDdDT = np.tensordot(dvect,dvect, axes=0)[:dimen, :dimen]
        return D - dDdDT/denom
        

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
    
    def calculate_Dp(self, elast, poiss, avect):
        D = self.calculate_D()
        dimen = D.shape[0]
        dvect = self.get_flowpl(elast, poiss, avect)
        denom = self.mater.hards + dvect@avect
        dDdDT = np.tensordot(dvect,dvect, axes=0)[:dimen, :dimen]
        return D - dDdDT/denom
    



def get_const_model(problem_type):
    problem_types = {'PlaneStress': PlaneStress, 
                     'PlaneStrain': PlaneStrain,
                     'Axisymmetric': Axisymmetric}
    
    return problem_types.get(problem_type)

#'''