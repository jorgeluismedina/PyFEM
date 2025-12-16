
import numpy as np
import scipy as sp
from .base_elem import FrameElement

class Frame2D(FrameElement):
    def __init__(self, mater, section, coord, conec, dof): #elast, xarea, i_mom)
        super().__init__(mater, section, coord, conec, dof)
        self.init_element()

    def init_element(self):
        vector = self.coord[1] - self.coord[0]
        self.length = sp.linalg.norm(vector)
        self.dirvec = vector/self.length

        c, s = self.dirvec
        c2 = c*c
        s2 = s*s
        cs = c*s
        
        EA = self.mater.elast * self.section.xarea
        EI = self.mater.elast * self.section.inrt3
        oneEA = EA / self.length
        twoEI = 2 * EI / self.length
        fourEI = 4 * EI / self.length
        sixEI = 6 * EI / self.length**2
        twelveEI = 12 * EI / self.length**3
        
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
        
        self.loads = np.zeros(6)
        self.forces = np.zeros(6)
  
    def add_loads(self, q1i, q2i, q1j, q2j):
        # Añadir en coordenadas locales
        # q1i = intensidad en el nodo i en direccion de la barra
        # q1j = intensidad en el nodo j en direccion de la barra
        # q2i = intensidad en el nodo i en direccion perpendicular de la barra
        # q2j = intensidad en el nodo j en direccion perpendicular de la barra
        
        qi_qj = q2i - q2j
        f1i =  (q1i/3 + q1j/6) * self.length
        f2i = -(3/20 * (qi_qj) - q2i/2) * self.length
        mi  = -(1/30 * (qi_qj) - q2i/12) * self.length**2

        f1j =  (q1j/3 + q1i/6) * self.length
        f2j = -(7/20 * (qi_qj) - q2i/2) * self.length
        mj  =  (1/20 * (qi_qj) - q2i/12) * self.length**2

        c, s = self.dirvec
        #self.loads = np.empty(6, dtype=float)
        self.loads[0] =  f1i*c - f2i*s
        self.loads[1] =  f1i*s + f2i*c
        self.loads[2] =  mi
        self.loads[3] =  f1j*c - f2j*s
        self.loads[4] =  f1j*s + f2j*c
        self.loads[5] =  mj

    
    def calculate_forces(self, glob_disps):
        forces = self.stiff @ glob_disps # global
        c, s = self.dirvec
        # Fuerzas a coordenadas locales
        #self.forces = np.empty(6, dtype=float)
        self.forces[0] = forces[0]*c + forces[1]*s
        self.forces[1] = -forces[0]*s + forces[1]*c
        self.forces[2] = forces[2]
        self.forces[3] = forces[3]*c + forces[4]*s
        self.forces[4] = -forces[3]*s + forces[4]*c
        self.forces[5] = forces[5]
        self.forces = self.forces - self.loads