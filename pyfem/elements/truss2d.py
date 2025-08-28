
import numpy as np
import scipy as sp
from .base_elem import Element

class Truss2D(Element):
    def __init__(self, nodes, coord, section, mater):
        super().__init__(nodes, coord, section, mater)
        self.set_dof(2)
        vector = self.coord[1] - self.coord[0]
        self.length = sp.linalg.norm(vector)
        self.dirvec = vector/self.length
        self.stress = 0.0
        self.yielded = False
        self.init_element()
    
    def init_element(self):
        c, s = self.dirvec
        cc = c*c
        ss = s*s
        cs = c*s
        EA_L = self.mater.elast * self.section.xarea / self.length
        self.stiff = EA_L * np.array([[cc, cs, -cc, -cs],
                                      [cs, ss, -cs, -ss],
                                      [-cc, -cs, cc, cs],
                                      [-cs, -ss, cs, ss]])
        
        pAL = self.mater.dense * self.section.xarea * self.length
        self.mass = pAL/6 * np.array([[2*cc, 2*cs, cc, cs],
                                      [2*cs, 2*ss, cs, ss],
                                      [cc, cs, 2*cc, 2*cs],
                                      [cs, ss, 2*cs, 2*ss]])

    def calculate_forces(self, glob_disps):
        EA_L = self.mater.elast * self.section.xarea / self.length
        c, s = self.dirvec
        B = np.array([-c, -s, c, s])
        self.force = EA_L * B @ glob_disps
    
    def calc_stress(self, glob_disps):
        E_L = self.elast / self.length
        c, s = self.dirvec
        B = np.array([-c, -s, c, s])
        return E_L * B @ glob_disps

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