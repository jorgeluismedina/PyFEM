
import numpy as np
import scipy as sp
from .base_elem import FrameElement

class Frame2D(FrameElement):
    def __init__(self, mater, section, coord, conec, dof): #elast, xarea, i_mom)
        super().__init__(mater, section, coord, conec, dof)
        vector = self.coord[1] - self.coord[0]
        self.length = sp.linalg.norm(vector)
        self.dirvec = vector/self.length
        self.init_element()

    def init_element(self):
        c, s = self.dirvec
        c2 = c*c
        s2 = s*s
        cs = c*s
        EA = self.mater.elast * self.section.xarea
        EI = self.mater.elast * self.section.inrt3
        oneEA = EA / self.length
        twoEI = 2 * EI / self.length
        fourEI = 4 * EI / self.length
        twelveEI = 12 * EI / self.length**3
        sixEI = 6 * EI / self.length**2

        k11 = oneEA*c2 + twelveEI*s2
        k22 = oneEA*s2 + twelveEI*c2
        k33 = fourEI
        k12 = (oneEA - twelveEI)*cs
        k13 = -sixEI*s
        k14 = -(oneEA*c2 + twelveEI*s2)
        k36 = twoEI
        k23 = sixEI*c

        self.stiff = np.array([[ k11,  k12,  k13,  k14, -k12,  k13],
                               [ k12,  k22,  k23, -k12, -k22,  k23],
                               [ k13,  k23,  k33, -k13, -k23,  k36],
                               [ k14, -k12, -k13,  k11,  k12, -k13],
                               [-k12, -k22, -k23,  k12,  k22, -k23],
                               [ k13,  k23,  k36, -k13, -k23,  k33]])
        
  

    def add_loads(self, fui, fvi, mi, fuj, fvj, mj): # Darle el signo de carga antes
        #u = normal
        #v = tangente
        c, s = self.dirvec
        # Cargas en coordenadas globales
        self.loads = np.array([fui*c + fvi*s, -fui*s + fvi*c, mi,
                               fuj*c + fvj*s, -fuj*s + fvj*c, mj])
    
    def add_loadss(self, qui, qvi, quj, qvj):
        # Funciona pero ahora probar con casos qi>qj,  etc.
        fui = -(qui/3 + quj/6) * self.length
        fvi =  (3/20)*(qvj-qvi)*self.length + qvi*self.length/2
        mi  =  (qvj-qvi)*self.length**2/30 + qvi*self.length**2/12

        fuj = -(quj/3 + qui/6) * self.length
        fvj =  (7/20)*(qvj-qvi)*self.length + qvi*self.length/2
        mj  = -((qvj-qvi)*self.length**2/20 + qvi*self.length**2/12)

        c, s = self.dirvec
        self.loads = np.array([fui*c + fvi*s, -fui*s + fvi*c, mi,
                               fuj*c + fvj*s, -fuj*s + fvj*c, mj])

    
    def calculate_forces(self, glob_disps):
        forces = self.stiff @ glob_disps
        c, s = self.dirvec
        # Fuerzas en coordenadas locales
        fui =  forces[0]*c + forces[1]*s
        fvi = -forces[0]*s + forces[1]*c
        fuj =  forces[3]*c + forces[4]*s
        fvj = -forces[3]*s + forces[4]*c

        self.force = np.array([fui, fvi, forces[2], fuj, fvj, forces[5]])