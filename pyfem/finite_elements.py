
import numpy as np
import scipy as sp
from scipy import linalg
from abc import ABC, abstractmethod
from .gauss_quad import Gauss_Legendre
from .shape_funcs import Node4Shape


class FElement(ABC):
    def __init__(self, nodes, coord, mater): 
        self.coord = coord
        self.nodes = nodes
        self.mater = mater
        self.nnods = nodes.shape[0]
        self.eload = None
        self.dof = None
        
    def set_dof(self, ndofn: int) -> None:
        self.ndofn = ndofn
        node_dof = np.arange(ndofn, dtype=int)
        self.dof = np.repeat(ndofn*self.nodes, ndofn) + np.tile(node_dof, self.nnods)
    
    @abstractmethod
    def set_stiff_mat(self):
        pass

    @abstractmethod
    def calc_stress(self, u):
        pass

    @abstractmethod
    def update_elem(self, stress):
        pass

    

class Quad4(FElement):
    def __init__(self, nodes, coord, thick, mater):
        super().__init__(nodes, coord, mater)
        self.set_dof(2)
        self.thick = thick
        self.shape = Node4Shape(ndofn=2)
        self.quad_scheme = Gauss_Legendre(2, ndim=2)
        self.set_stiff_mat()

    def get_strain_mat(self, r, s):
        deriv = self.shape.deriv(r,s)
        jacob = deriv @ self.coord
        cartd = sp.linalg.inv(jacob) @ deriv
        B = np.zeros((3,8)) #8=nnods*ndofn
        B[0, 0::2] = cartd[0, :]
        B[1, 1::2] = cartd[1, :]
        B[2, 0::2] = cartd[1, :]
        B[2, 1::2] = cartd[0, :]
        return B, linalg.det(jacob)

    def set_stiff_mat(self):
        npoin = self.quad_scheme.npoin
        self.stress = np.zeros((npoin,3))
        self.yielded = np.zeros(npoin, dtype=bool)
        self.dmatx = self.mater.calculate_dmatx(npoin)
        self.bmatx = np.zeros((npoin,3,8))
        self.nmatx = np.zeros((npoin,2,8))
        bforc = np.array([0, -self.mater.dense])
        det_J = np.zeros(npoin)

        for i, point in enumerate(self.quad_scheme.points):
            self.nmatx[i] = self.shape.matrix(*point)
            self.bmatx[i], det_J[i] = self.get_strain_mat(*point)

        bmatx_T = np.transpose(self.bmatx,(0,2,1))
        self.dvolu = self.thick * det_J * self.quad_scheme.weights
        self.stiff = np.sum(bmatx_T @ self.dmatx @ self.bmatx * self.dvolu[:,None,None], axis=0)
        self.eload = np.sum(bforc @ self.nmatx * self.dvolu[:,None], axis=0)

    def calc_stress(self, u):
        #Estas partes no importa de momento
        #poins = 1/self.quad_scheme.points
        #gvals = self.shape.funcs(*poins.T).T #4x4
        #order = [0,2,3,1]
        gauss_stress = self.dmatx @ self.bmatx @ u #4x3
        #nodes_stress = gvals[order] @ gauss_stress[order] #4x3
        return gauss_stress#, nodes_stress

    def update_elem(self, delta_stress):
        self.stress += delta_stress
        yield_crite = self.mater.yield_crite
        modi_stress = self.mater.const_model.prepare_stress(self.stress)
        prev_yielded = np.copy(self.yielded)
        for i, stress in enumerate(modi_stress):
            yield_crite.enter_stress(stress)
            if yield_crite.check_yield():
                if not self.yielded[i]:
                    avect = yield_crite.get_flow_vector()
                    self.dmatx[i] = self.mater.calculate_Dp(avect)
                    self.yielded[i] = True

        #return prev_yielded, self.yielded
        if not np.array_equal(prev_yielded, self.yielded):
            bmatx_T = np.transpose(self.bmatx,(0,2,1))
            self.stiff = np.sum(bmatx_T @ self.dmatx @ self.bmatx * self.dvolu[:,None,None], axis=0)
        #return self.stiff

    

    

class Bar1D(FElement):
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

        


    
    
#Cambiar esta clase  
class Bar2D(FElement):
    def __init__(self, nodes, coord, xarea, mater):
        super().__init__(nodes, coord, mater)
        self.set_dof(2)
        vector = self.coord[1] - self.coord[0]
        self.xarea = xarea
        self.elast = self.mater[0]
        self.length = sp.linalg.norm(vector)
        self.dirvec = vector/self.length
        self.get_stiff_mat()
        self.get_mass_mat()
    
    def rotation_matrix(self):
        c, s = self.dirvec
        R = np.array([[c, s, 0, 0], 
                      [0, 0, c, s]])
        return R
    
    def get_stiff_mat(self):
        EA_L = self.elast * self.xarea / self.length
        K = EA_L * np.array([[ 1, -1],
                             [-1,  1]])
        R = self.rotation_matrix()
        self.stiff = R.T @ K @ R

    def get_stress(self, u):
        E_L = self.elast / self.length
        c, s = self.dirvec
        B = np.array([-c, -s, c, s])
        self.stress = E_L * B @ u # E*B*u
        self.iforce = self.stress * self.xarea  

#class Beam2D()