
import numpy as np
import scipy as sp
from pyfem.elements.base_elem import AreaElement
from pyfem.gauss_quad import gauss_nd
from pyfem.shape_funcs import shape_quad8, deriv_quad8
from pyfem.materials.yield_criterion import StressState2D



#Esta clase tiene que variar para problemas axisimetricos ya que B es una matriz de (4,8)
class Quad8(AreaElement):
    def __init__(self, mater, section, coord, conec, dof):
        super().__init__(mater, section, coord, conec, dof)
        #self.yield_crite = self.mater.yield_crite
        self.init_element()
       
    def get_shape_mat(self, r, s):
        shape = shape_quad8(r,s)
        N = np.zeros((2,16)) #(2,2*nnods)
        N[0, 0::2] = shape
        N[1, 1::2] = shape
        return N 
    
    def get_strain_mat(self, r, s):
        deriv = deriv_quad8(r,s)
        jacob = deriv @ self.coord
        cartd = sp.linalg.inv(jacob) @ deriv
        B = np.zeros((3,16)) #(3,2*nnods)
        B[0, 0::2] = cartd[0, :]
        B[1, 1::2] = cartd[1, :]
        B[2, 0::2] = cartd[1, :]
        B[2, 1::2] = cartd[0, :]
        return B, sp.linalg.det(jacob)
    
    
    
    def get_stiff_mat(self):
        bmatx_T = np.transpose(self.bmatx,(0,2,1)) #Transponer B por cada punto de integracion
        BT_D_B = bmatx_T @ self.dmatx @ self.bmatx 
        return np.sum(BT_D_B * self.darea[:,None,None] * self.section.thick, axis=0)
    

    def init_element(self):
        points, weights = gauss_nd(2,2)
        npoin = points.shape[0]
        #self.stress = np.zeros((npoin,3))
        self.stress = StressState2D()
        #self.yielded = np.zeros(npoin, dtype=bool)
        self.dmatx = np.tile(self.mater.dmatx, (npoin,1,1)) # D = [(3,3),(3,3),(3,3),(3,3)]
        self.bmatx = np.zeros((npoin,3,16)) # B = [(3,16),(3,16),(3,16),(3,16)]
        self.nmatx = np.zeros((npoin,8)) # N = [(2,16),(2,16),(2,16),(2,16)]
        det_J = np.zeros(npoin)

        for i, point in enumerate(points):
            self.nmatx[i] = shape_quad8(*point)
            self.bmatx[i], det_J[i] = self.get_strain_mat(*point)
        
        self.area = np.dot(det_J, weights)
        self.darea = det_J * weights
        self.stiff = self.get_stiff_mat()
        #self.bload = self.get_body_load()
        self.loads = np.zeros(16)
        self.add_selfweight()

    def add_selfweight(self):
        Nb = -self.mater.dense * self.section.thick
        Nb_nodal = np.sum(Nb * self.nmatx * self.darea[:,None], axis=0)
        self.loads[1::2] += Nb_nodal
        #bforc = np.array([0, -self.mater.dense * self.section.thick])
        #N_b = bforc @ self.nmatx 
        #return selfweight#np.sum(N_b * self.darea[:,None], axis=0)

    def calculate_stress(self, glob_disps):
        #siguen habiendo 4 puntos de integracion
        gauss_stress = self.dmatx @ self.bmatx @ glob_disps #4x3
        self.stress.update(gauss_stress[:,0], 
                           gauss_stress[:,1], 
                           gauss_stress[:,2])
        #return gauss_stress#, nodes_stress