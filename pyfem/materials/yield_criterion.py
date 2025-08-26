
import numpy as np
from abc import ABC, abstractmethod
"""
    Calculate principal stresses from normal stresses using Mohr's circle formulas.

    Args:
    - normal_stresses (numpy.ndarray): Array of normal stresses with shape (3, n) where n is the number of integration points.
                                        Each column contains normal stresses [sigma_x, sigma_y, tau_xy] at an integration point.

    Returns:
    - principal_stresses (numpy.ndarray): Array of principal stresses with shape (3, n) where n is the number of integration points.
                                          Each column contains principal stresses [sigma1, sigma2, dir] at an integration point.
    """
def principal_stresses_mohr(normal_stresses):
    n = normal_stresses.shape[1]
    principal_stresses = np.zeros((3, n))
    for i in range(n):
        sigma_x, sigma_y, tau_xy = normal_stresses[:, i]

        # Calculate principal stresses
        sigma_avg = (sigma_x + sigma_y) / 2
        delta_sigma = np.sqrt(((sigma_x - sigma_y) / 2)**2 + tau_xy**2)
        sigma1 = sigma_avg + delta_sigma
        sigma2 = sigma_avg - delta_sigma

        # Calculate principal direction
        if tau_xy != 0:
            theta_p = np.arctan2(2 * tau_xy, sigma_x - sigma_y) / 2
        else:
            theta_p = 0

        principal_stresses[:, i] = [sigma1, sigma2, theta_p]

    return principal_stresses 


class StressState2D():
    def __init__(self, sx, sy, txy, sz):
        self.sx = sx
        self.sy = sy
        self.txy = txy
        self.sz = sz
        #self.hi= 


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

    @abstractmethod
    def get_constants(self):
        pass



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