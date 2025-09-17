
import numpy as np
from abc import ABC, abstractmethod


class StressState2D():
    def __init__(self, sx=0.0, sy=0.0, txy=0.0):
        self.sx = sx
        self.sy = sy
        self.sz = 0.0
        self.txy = txy
        self.calculate_properties()
        
    def calculate_properties(self):
        self.calculate_principal()
        self.calculate_hydrostatic()
        self.calculate_deviatoric()

    def update(self, sx=None, sy=None, txy=None):
        if sx is not None:
            self.sx = np.asarray(sx)
        if sy is not None:
            self.sy = np.asarray(sy)
        if txy is not None:
            self.txy = np.asarray(txy)
        self.calculate_properties()
        
    
    def calculate_principal(self):
        smean = (self.sx + self.sy)/2
        sdelt = np.sqrt((self.sx - self.sy)**2/4 + self.txy**2)
        s1 = smean + sdelt
        s2 = smean - sdelt
        self.s1 = np.where(s1>s2, s1, s2)
        self.s2 = np.where(s1<s2, s1, s2)
        self.s3 = np.zeros_like(self.s1)
        # Cálculo vectorizado del ángulo principal
        denominator = self.sx - self.sy
        # Evitar división por cero
        # Para casos donde sx = sy, establecemos el ángulo en 45°
        self.direc = np.where(
            np.abs(denominator) > 1e-10,  # Condición para evitar división por cero
            0.5 * np.arctan2(2 * self.txy, denominator),
            np.pi/4)  # Valor cuando sx = sy
    
    def calculate_hydrostatic(self):
        self.hidro = (self.s1 + self.s2 + self.s3)/3

    def calculate_deviatoric(self):
        self.s1_dev = self.s1 - self.hidro
        self.s2_dev = self.s2 - self.hidro
        self.s3_dev = self.s3 - self.hidro

    def get_cartesians(self):
        return np.array([self.sx, self.sy, self.txy])
    
    def get_principals(self):
        return np.array([self.s1, self.s2, self.s3])
    
    def get_von_mises(self):
        return np.sqrt(self.s1**2 - self.s1*self.s2 + self.s2**2)




class YieldCriterion(ABC):
    def __init__(self, material):
        self.root3 = np.sqrt(3)
        self.mater = material
        #self.uniax = self.mater.uniax if hasattr(self.mater, 'uniax') else None

    def get_theta(self):
        J3 = self.devia[3]*(self.devia[3]**2-self.J2)
        if self.J2==0:
            sin3t = 0
        else:
            value = -3*self.root3*J3 / (2*self.J2*self.steff)
            sin3t = -1 if value < -1 else 1 if value > 1 else value
        return np.arcsin(sin3t)/3
         
    def calc_invariants(self, stress): #esta funcion necesita ser ejecutada antes hacer todo
        trace = np.array([True, True, False, True])
        self.smean = np.sum(stress[trace])/3 #I3/3
        self.devia = stress - self.smean*trace
        self.J2 = self.devia[2]**2 + 0.5*np.sum(self.devia[trace]**2)
        self.steff = np.sqrt(self.J2)
        self.theta = self.get_theta()
    
    def get_flow_vector(self):
        veca1 = np.array([1,1,0,1])
        veca2 = self.devia*np.array([1,1,2,1])/(2*self.steff)
        veca3 = np.array([self.devia[1]*self.devia[3] + self.J2/3,
                          self.devia[0]*self.devia[3] + self.J2/3,
                          -2*self.devia[2]*self.devia[3],
                          self.devia[0]*self.devia[1] - self.devia[2]**2 + self.J2/3])
        
        cons1, cons2, cons3 = self.get_constants()
        avect = cons1*veca1 + cons2*veca2 + cons3*veca3
        return avect


    def enter_stress(self, stress): #[sx, sy, tyx, sz]
        self.calc_invariants(stress)

    def check_yield(self):
        steff = self.stress_level()
        return steff > self.mater.uniax
        
    @abstractmethod
    def stress_level(self): #(effective stress)
        pass

    #@abstractmethod
    #def get_constants(self):
    #    pass



class Tresca(YieldCriterion):
    def stress_level(self):
        costh = np.cos(self.theta)
        return 2*costh*self.steff
    
    def get_constants(self):
        cons1 = 0
        abthe = abs(self.theta/np.pi*180) #condicion en deg
        if abthe<30:
            costh = np.cos(self.theta)
            sinth = np.sin(self.theta)
            tan3t = np.tan(3*self.theta)
            cos3t = np.cos(3*self.theta)
            cons2 = 2*(costh + sinth*tan3t)
            cons3 = self.root3*sinth/(self.J2*cos3t)
        else:
            cons2 = self.root3
            cons3 = 0
        return cons1, cons2, cons3

    
class VonMises(YieldCriterion):
    def stress_level(self):
        return self.root3*self.steff
    
    def get_constants(self):
        cons1 = 0
        cons2 = self.root3
        cons3 = 0
        return cons1, cons2, cons3

class MohrCoulomb(YieldCriterion):
    def __init__(self, material):
        super().__init__(material)
        self.phira = self.mater.frict * np.pi/180 #en rad
        self.snphi = np.sin(self.phira)
        self.uniax = self.mater.cohes * np.cos(self.phira)

    def stress_level(self):
        costh = np.cos(self.theta)
        sinth = np.sin(self.theta)
        expr1 = costh - sinth*self.snphi/self.root3  
        return self.smean*self.snphi + self.steff*expr1
    
    def get_constants(self):
        cons1 = self.snphi/3
        abthe = abs(self.theta/np.pi*180) #condicion en deg
        if abthe<30:
            costh = np.cos(self.theta)
            sinth = np.sin(self.theta)
            tanth = np.tan(self.theta)
            tan3t = np.tan(3*self.theta)
            cos3t = np.cos(3*self.theta)
            expr1 = cons1*(tan3t-tanth)*self.root3
            expr2 = self.root3*sinth + 3*cons1*costh
            cons2 = costh*(1+tanth*tan3t+expr1)
            cons3 = expr2/(2*self.J2*cos3t)
        else:
            plumi = -1 if self.theta>0 else 1
            cons2 = 0.5*(self.root3 + plumi*cons1*self.root3)
            cons3 = 0

        return cons1, cons2, cons3

class DruckerPrager(YieldCriterion):
    def __init__(self, material):
        super().__init__(material)
        self.phira = self.mater.frict * np.pi/180 #en rad
        self.snphi = np.sin(self.phira)
        self.uniax = (6*self.mater.cohes*np.cos(self.phira)) / (self.root3*(3-self.snphi))   
    
    def stress_level(self):
        expr1 = 6*self.smean*self.snphi/(self.root3*(3-self.snphi))
        return expr1 + self.steff
    
    def get_constants(self):
        cons1 = 2*self.snphi/(self.root3*(3-self.snphi))
        cons2 = 1
        cons3 = 0
        return cons1, cons2, cons3



def get_yield_criterion(criterion):
    yield_criteria = {'Tresca': Tresca, 
                      'VonMises': VonMises,
                      'MohrCoulomb': MohrCoulomb,
                      'DruckerPrager': DruckerPrager}
    
    return yield_criteria.get(criterion)