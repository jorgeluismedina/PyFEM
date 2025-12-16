

import numpy as np
import scipy as sp
from .base_elem import FrameElement

class Frame22D(FrameElement):
    def __init__(self, mater, section, coord, conec, dof):
        super().__init__(mater, section, coord, conec, dof)
        vector = self.coord[1] - self.coord[0]
        self.length = sp.linalg.norm(vector)
        self.dirvec = vector/self.length
        self.init_element()

    def rmatx(self):
        c, s = self.dirvec
        rmatx = np.eye(6)
        rmatx[0:2, 0:2] = np.array([[c, s],[-s, c]])
        rmatx[3:5, 3:5] = np.array([[c, s],[-s, c]])

        return rmatx

    def kmatx(self):
        EA = self.mater.elast * self.section.xarea
        EI = self.mater.elast * self.section.inrt3
        oneEA = EA / self.length
        twoEI = 2 * EI / self.length
        fourEI = 4 * EI / self.length
        twelveEI = 12 * EI / self.length**3
        sixEI = 6 * EI / self.length**2

        kmatx = np.zeros((6,6))
        kmatx[0, [0,3]] = [ oneEA, -oneEA]
        kmatx[1, [1,2,4,5]] = [ twelveEI,  sixEI,   -twelveEI,  sixEI]
        kmatx[2, [1,2,4,5]] = [ sixEI,     fourEI,  -sixEI,     twoEI]
        kmatx[3, [0,3]] = [-oneEA, oneEA]
        kmatx[4, [1,2,4,5]] = [-twelveEI, -sixEI,    twelveEI,  -sixEI]
        kmatx[5, [1,2,4,5]] = [ sixEI,     twoEI,   -sixEI,      fourEI]

        return kmatx
    
    def mmatx(self):
        pA = self.mater.dense * self.section.xarea
        L = self.length

        mmatx = np.zeros((6,6))
        mmatx[0, [0,3]] = [ 140, 70]
        mmatx[1, [1,2,4,5]] = [ 156,  22*L,  54,  -13*L]
        mmatx[2, [1,2,4,5]] = [ 22*L, 4*L**2,  13*L,  -3*L**2]
        mmatx[3, [0,3]] = [70, 140]
        mmatx[4, [1,2,4,5]] = [54, 13*L, 156,  -22*L]
        mmatx[5, [1,2,4,5]] = [-13*L, -3*L**2, -22*L, 4*L**2]
        return pA*L/420 * mmatx


    
    def init_element(self):
        self.stiff = self.rmatx().T @ self.kmatx() @ self.rmatx()
        self.mass = self.rmatx().T @ self.mmatx() @ self.rmatx()
        self.loads = np.zeros(6)
        self.forces = np.zeros(6)

    def add_loadss(self, q1i, q2i, q1j, q2j):
        # Añadir en coordenadas locales
        # q1i = intensidad en el nodo i en direccion de la barra
        # q1j = intensidad en el nodo j en direccion de la barra
        # q2i = intensidad en el nodo i en direccion perpendicular de la barra
        # q2j = intensidad en el nodo j en direccion perpendicular de la barra
        qi_qj = q2i - q2j

        self.loads[0] =  (q1i/3 + q1j/6) * self.length
        self.loads[1] = -(3/20 * (qi_qj) - q2i/2) * self.length
        self.loads[2]  = -(1/30 * (qi_qj) - q2i/12) * self.length**2

        self.loads[3] =  (q1j/3 + q1i/6) * self.length
        self.loads[4] = -(7/20 * (qi_qj) - q2i/2) * self.length
        self.loads[5]  =  (1/20 * (qi_qj) - q2i/12) * self.length**2

        self.loads = self.rmatx() @ self.loads
    
    def calculate_forces(self, glob_disps):
        # A Coordenadas locales
        self.forces = self.rmatx() @ self.stiff @ glob_disps
        self.forces = self.forces - self.loads

        

    def release_ends(self, ui=1, vi=1, ri=1, uj=1, vj=1, rj=1): # liberar es poner 0
        retained = np.array([ui, vi, ri, uj, vj, rj], dtype=bool)
        released = np.bitwise_not(retained)

        kmatx = self.kmatx()
        Knn = kmatx[np.ix_(retained, retained)]
        Knd = kmatx[np.ix_(retained, released)]
        Kdn = kmatx[np.ix_(released, retained)]
        Kdd = kmatx[np.ix_(released, released)]

        # Subvectores de fuerzas
        fn = self.loads[retained]
        fd = self.loads[released]
        
        X = np.linalg.solve(Kdd, Kdn)
        Y = np.linalg.solve(Kdd, fd)

        # Condensed matrices
        Knn_ = Knn - Knd @ X
        fn_  = fn - Knd @ Y

        k_mod = np.zeros_like(kmatx)
        k_mod[np.ix_(retained, retained)] = Knn_
        self.stiff = self.rmatx().T @ k_mod @ self.rmatx()

        f_mod = np.zeros_like(self.loads)
        f_mod[retained] = fn_
        self.loads = self.rmatx() @ f_mod