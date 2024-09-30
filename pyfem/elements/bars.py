
import numpy as np
import scipy as sp
from pyfem.elements.element_base import Element

class Bar1D(Element):
    def __init__(self, nodes, coord, xarea, mater): #mater= [E, nu, sy, Hp, dens]
        super().__init__(nodes, coord, mater)
        self.set_dof(1)
        self.xarea = xarea
        self.elast = self.mater.elast
        self.length = np.abs(self.coord[1] - self.coord[0])
        self.stress = 0.0
        self.yielded = False
        self.set_stiff_mat()
    
    def set_stiff_mat(self):
        EA_L = self.elast * self.xarea / self.length
        self.stiff = EA_L * np.array([[ 1, -1],
                                      [-1,  1]])

    def calc_stress(self, u):
        E_L = self.elast / self.length
        B = 1/self.length * np.array([-1, 1])
        return E_L * B @ u

    def update_elem(self, delta_stress):
        self.stress += delta_stress
        if abs(self.stress) > self.mater.uniax:
            if not self.yielded:
                elast = self.mater.elast
                hards = self.mater.hards
                facto = (1-elast/(elast+hards))
                self.elast = facto*self.elast
                self.stiff = facto*self.stiff
                self.yielded = True


class Bar2D(Element):
    def __init__(self, nodes, coord, xarea, mater):
        super().__init__(nodes, coord, mater)
        self.set_dof(2)
        vector = self.coord[1] - self.coord[0]
        self.xarea = xarea
        self.elast = self.mater.elast
        self.length = sp.linalg.norm(vector)
        self.dirvec = vector/self.length
        self.stress = 0.0
        self.yielded = False
        self.set_stiff_mat()
    
    def rotation_matrix(self):
        c, s = self.dirvec
        R = np.array([[c, s, 0, 0], 
                      [0, 0, c, s]])
        return R
    
    def set_stiff_mat(self):
        EA_L = self.elast * self.xarea / self.length
        K = EA_L * np.array([[ 1, -1],
                             [-1,  1]])
        R = self.rotation_matrix()
        self.stiff = R.T @ K @ R

    def calc_stress(self, u):
        E_L = self.elast / self.length
        c, s = self.dirvec
        B = np.array([-c, -s, c, s])
        return E_L * B @ u

    def update_elem(self, delta_stress):
        self.stress += delta_stress
        if abs(self.stress) > self.mater.uniax:
            if not self.yielded:
                elast = self.mater.elast
                hards = self.mater.hards
                facto = (1-elast/(elast+hards))
                self.elast = facto*self.elast
                self.stiff = facto*self.stiff
                self.yielded = True  