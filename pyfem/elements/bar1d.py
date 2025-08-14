
import numpy as np
import scipy as sp
from pyfem.elements.base_elem import Element

class Bar1D(Element):
    def __init__(self, nodes, coord, section, mater): #mater= [E, nu, sy, Hp, dens]
        super().__init__(nodes, coord, section, mater)
        self.set_dof(1)
        self.xarea = self.section.xarea
        self.elast = self.mater.elast
        self.length = np.abs(self.coord[1] - self.coord[0])
        self.stress = 0.0
        self.yielded = False
        self.init_element()
    
    def init_element(self):
        EA_L = self.elast * self.xarea / self.length
        self.stiff = EA_L * np.array([[ 1, -1],
                                      [-1,  1]])

    def calc_stress(self, disps):
        E_L = self.elast / self.length
        B = 1/self.length * np.array([-1, 1])
        return E_L * B @ disps

    def update_stiff(self, delta_stress):
        self.stress += delta_stress
        if abs(self.stress) > self.mater.uniax:
            if not self.yielded:
                elast = self.mater.elast
                hards = self.mater.hards
                facto = (1-elast/(elast+hards))
                self.elast = facto*self.elast
                self.stiff = facto*self.stiff
                self.yielded = True


  