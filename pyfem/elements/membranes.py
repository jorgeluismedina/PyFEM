
import numpy as np
import scipy as sp
from pyfem.elements.element_base import Element
from pyfem.gauss_quad import Gauss_Legendre
from pyfem.shape_funcs import Node4Shape, Node8Shape

#Esta clase tiene que variar para problemas axisimetricos ya que B es una matriz de (4,8)
class Membrane(Element):
    def __init__(self, nodes, coord, thick, mater):
        super().__init__(nodes, coord, mater)
        self.set_dof(2)
        self.thick = thick
        self.yield_crite = self.mater.yield_crite
        self.const_model = self.mater.const_model

    def get_strain_mat(self, r, s):
        deriv = self.shape.deriv(r,s)
        jacob = deriv @ self.coord
        cartd = sp.linalg.inv(jacob) @ deriv
        B = np.zeros((3,2*self.nnods)) #8=nnods*ndofn
        B[0, 0::2] = cartd[0, :]
        B[1, 1::2] = cartd[1, :]
        B[2, 0::2] = cartd[1, :]
        B[2, 1::2] = cartd[0, :]
        return B, sp.linalg.det(jacob)

    def set_stiff_mat(self):
        npoin = self.quad_scheme.npoin
        self.stress = np.zeros((npoin,3))
        self.yielded = np.zeros(npoin, dtype=bool)
        self.dmatx = self.mater.calculate_dmatx(npoin)
        self.bmatx = np.zeros((npoin,3,2*self.nnods))
        self.nmatx = np.zeros((npoin,2,2*self.nnods))
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
        prev_yielded = np.copy(self.yielded)
        modi_stress = self.const_model.all_components(self.stress)
        
        for i, stress in enumerate(modi_stress):
            self.yield_crite.enter_stress(stress)
            
            if self.yield_crite.check_yield():
                if not self.yielded[i]:
                    avect = self.yield_crite.get_flow_vector()
                    self.dmatx[i] = self.mater.calculate_Dp(avect)
                    self.yielded[i] = True

        if not np.array_equal(prev_yielded, self.yielded):
            bmatx_T = np.transpose(self.bmatx,(0,2,1))
            self.stiff = np.sum(bmatx_T @ self.dmatx @ self.bmatx * self.dvolu[:,None,None], axis=0)



class Quad4(Membrane):
    def __init__(self, nodes, coord, thick, mater):
        super().__init__(nodes, coord, thick, mater)
        self.shape = Node4Shape(ndofn=2)
        self.quad_scheme = Gauss_Legendre(2, ndim=2)
        self.set_stiff_mat()


class Quad8(Membrane):
    def __init__(self, nodes, coord, thick, mater):
        super().__init__(nodes, coord, thick, mater)
        self.shape = Node8Shape(ndofn=2)
        self.quad_scheme = Gauss_Legendre(2, ndim=2)
        self.set_stiff_mat()
