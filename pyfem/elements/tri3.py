

import numpy as np
import scipy as sp
from pyfem.elements.base_elem import Element
from pyfem.gauss_quad import Gauss_Legendre
from pyfem.shape_funcs import ShapeTri3



#Esta clase tiene que variar para problemas axisimetricos ya que B es una matriz de (4,8)
class Tri3(Element):
    def __init__(self, nodes, coord, section, mater): # O que reciba un Gauss_Legendre o un Node4Shape para evitar instancias repetidas
        super().__init__(nodes, coord, section, mater)
        self.set_dof(2)
        #self.yield_crite = self.mater.yield_crite
        #self.const_model = self.mater.const_model
        self.shape = ShapeTri3() # Que esta clase sean funciones de esta misma clase para ahorrar memoria (instancias repetidas)
        #self.quad_scheme = Gauss_Legendre(2, ndim=2) # Que Gauss_Legendre solo sea una funcion para ahorrar (instancias repetidas)
        self.init_element() 
    
    #def get_shape_mat(self, r, s):
    #    shape = self.shape.funcs(r,s)
    #    N = np.zeros((2,8)) #(2,2*nnods)
    #    N[0, 0::2] = shape
    #    N[1, 1::2] = shape
    #    return N 
    
    def get_strain_mat(self):
        #deriv = self.shape.deriv(r,s)
        #jacob = deriv @ self.coord
        #cartd = sp.linalg.inv(jacob) @ deriv
        b1 = self.coord[1,1] - self.coord[2,1]
        b2 = self.coord[2,1] - self.coord[0,1]
        b3 = self.coord[0,1] - self.coord[1,1]
        c1 = self.coord[2,0] - self.coord[1,0]
        c2 = self.coord[0,0] - self.coord[2,0]
        c3 = self.coord[1,0] - self.coord[0,0]
        area2 = np.abs(b1*c2 - b2*c1)
        area = area2 / 2
        B = np.zeros((3,6)) #(3,2*nnods)
        B[0, 0::2] = [b1, b2, b3]
        B[1, 1::2] = [c1, c2, c3]
        B[2, 0::2] = [c1, c2, c3]
        B[2, 1::2] = [b1, b2, b3]
        B = B / area2
        return B, area
    
    def get_body_load(self):
        bload = np.zeros(6)
        bforc = self.area * self.mater.dense * self.section.thick/3
        bload[1::2] = bforc
        return bload
    
    def get_stiff_mat(self):
        BT_D_B = self.bmatx.T @ self.dmatx @ self.bmatx
        return BT_D_B * self.area * self.section.thick
    
    def init_element(self):
        #npoin = self.quad_scheme.npoin
        self.stress = np.zeros(3)
        self.yielded = False
        # modificar
        const = self.mater.elast/(1-self.mater.poiss**2)
        conss = (1-self.mater.poiss)/2
        dmatx = const*np.array([[1, self.mater.poiss, 0],
                                [self.mater.poiss, 1, 0],
                                [0, 0, conss]])
        self.dmatx = dmatx
        #-------------------------------------------
        self.bmatx, self.area = self.get_strain_mat() # B = [(3,8),(3,8),(3,8),(3,8)]
        #self.nmatx = np.zeros((npoin,2,8)) # N = [(2,8),(2,8),(2,8),(2,8)]
        self.stiff = self.get_stiff_mat()
        self.bload = self.get_body_load()

    def calculate_stress(self, glob_disps):
        #Estas partes no importa de momento
        #poins = 1/self.quad_scheme.points
        #gvals = self.shape.funcs(*poins.T).T #4x4
        #order = [0,2,3,1]
        stress = self.dmatx @ self.bmatx @ glob_disps #4x3
        #nodes_stress = gvals[order] @ gauss_stress[order] #4x3
        self.stress = stress#, nodes_stress