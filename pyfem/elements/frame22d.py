

import numpy as np
import scipy as sp
from .base_elem import Element

class Frame22D(Element):
    def __init__(self, nodes, coord, section, mater):
        super().__init__(nodes, coord, section, mater)
        self.set_dof(3)
        vector = self.coord[1] - self.coord[0]
        self.length = sp.linalg.norm(vector)
        self.dirvec = vector/self.length
        self.init_element()
        self.force = None

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

    
    def init_element(self):
        self.stiff = self.rmatx().T @ self.kmatx() @ self.rmatx()
        self.loads = np.zeros(6)

    def add_loadss(self, qui, qvi, quj, qvj):
        fui = -(qui/3 + quj/6) * self.length
        fvi =  (3/20)*(qvj-qvi)*self.length + qvi*self.length/2
        mi  =  (qvj-qvi)*self.length**2/30 + qvi*self.length**2/12

        fuj = -(quj/3 + qui/6) * self.length
        fvj =  (7/20)*(qvj-qvi)*self.length + qvi*self.length/2
        mj  = -((qvj-qvi)*self.length**2/20 + qvi*self.length**2/12)

        self.loads = self.rmatx() @ np.array([fui, fvi, mi, fuj, fvj, mj])
    
    def calculate_forces(self, glob_disps):
        # A Coordenadas locales
        self.force = self.rmatx() @ self.stiff @ glob_disps

        

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