
import numpy as np
import scipy as sp
from scipy import linalg
from .gauss_quad import Gauss_Legendre
from .shape_funcs import Node4Shape


def const_mat_2d(E, nu):
    enu = E / (1-nu**2)
    mnu = (1-nu)/2
    D = enu*np.array([[1, nu, 0],
                      [nu, 1, 0],
                      [0, 0, mnu]])
    return D


class FElement():
    def __init__(self, nodes, coord, param, mater): 
        self.coord = coord
        self.nodes = nodes
        self.mater = mater
        self.param = param
        self.nnods = nodes.shape[0]
        self.eload = None
        self.dof = None
        
    def set_dof(self, ndofn: int) -> None:
        self.ndofn = ndofn
        node_dof = np.arange(ndofn, dtype=int)
        self.dof = np.repeat(ndofn*self.nodes, ndofn) + np.tile(node_dof, self.nnods)

    

class Quad4(FElement):
    def __init__(self, nodes, coord, param, mater):
        super().__init__(nodes, coord, param, mater)
        self.set_dof(2)
        self.thick = self.param
        self.shape = Node4Shape(ndofn=2)
        self.dmatx = const_mat_2d(self.mater[0], self.mater[1])
        self.quad_scheme = Gauss_Legendre(2, ndim=2)
        self.set_stiff_mat()
        self.cracked = False
    
    def get_strain_mat(self, r, s):
        deriv = self.shape.deriv(r,s)
        jacob = deriv @ self.coord
        cartd = sp.linalg.inv(jacob) @ deriv
        B = np.zeros((3,8))
        B[0, 0::2] = cartd[0, :]
        B[1, 1::2] = cartd[1, :]
        B[2, 0::2] = cartd[1, :]
        B[2, 1::2] = cartd[0, :]
        return B, linalg.det(jacob)

    def set_stiff_mat(self):
        npoin = self.quad_scheme.points.shape[0]
        bforc = np.array([0, -self.mater[-1]])
        bmatx = np.zeros((npoin,3,8))
        nmatx = np.zeros((npoin,2,8))
        det_J = np.zeros(npoin)

        for i, point in enumerate(self.quad_scheme.points):
            nmatx[i] = self.shape.matrix(*point)
            bmatx[i], det_J[i] = self.get_strain_mat(*point)

        bmatx_T = np.transpose(bmatx,(0,2,1))
        dvolu = self.thick * det_J * self.quad_scheme.weights
        self.stiff = np.sum(bmatx_T @ self.dmatx @ bmatx * dvolu[:,None,None], axis=0)
        self.eload = np.sum(bforc @ nmatx * dvolu[:,None], axis=0)
        self.bmatx = bmatx
        self.dvolu = dvolu

    def update_stiff_mat(self):
        bmatx_T = np.transpose(self.bmatx,(0,2,1))
        dvolu = self.dvolu[:,None,None]
        self.stiff = np.sum(bmatx_T @ self.dmatx @ self.bmatx * dvolu, axis=0)
            
    def get_stress(self, u):
        #self.stress = np.zeros((4*3))
        poins = 1/self.quad_scheme.points
        gvals = self.shape.funcs(*poins.T).T #4x4
        order = [0,2,3,1]
        gauss_stress = self.dmatx @ self.bmatx @ u #4x3
        nodes_stress = gvals[order] @ gauss_stress[order] #4x3
        return gauss_stress, nodes_stress




class Bar2D(FElement):
    def __init__(self, nodes, coord, param, mater):
        super().__init__(nodes, coord, param, mater)
        self.set_dof(2)
        vector = self.coord[1] - self.coord[0]
        self.xarea = self.param
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
    

    

class Bar1D(FElement):
    def __init__(self, nodes, coord, param, mater): #mater= [E, nu, sy, Hp, dens]
        super().__init__(nodes, coord, param, mater)
        self.set_dof(1)
        self.xarea = self.param
        self.elast = self.mater[0]
        self.length = np.abs(self.coord[1] - self.coord[0])
        self.get_stiff_mat()
        self.stress = 0.0
        self.yielded = False
    
    def get_stiff_mat(self):
        EA_L = self.elast * self.xarea / self.length
        K = EA_L * np.array([[ 1, -1],
                             [-1,  1]])
        self.stiff = K

    def mod_stiff_mat(self):
        E = self.elast
        Hp = self.mater[3]
        fact = (1-E/(E+Hp))
        self.elast = fact * self.elast
        self.stiff = fact * self.stiff

    def get_stress(self, u):
        sy = self.mater[2]
        E_L = self.elast / self.length
        B = 1/self.length * np.array([-1, 1])
        self.stress += E_L * B @ u
        if abs(self.stress) > sy:
            if not self.yielded:
                self.mod_stiff_mat()
                self.yielded = True
    
    
    

#class Beam2D()