
import numpy as np
import scipy as sp
from pyfem.elements.base_elem import AreaElement
from pyfem.gauss_quad import gauss_nd
from pyfem.shape_funcs import shape_quad8, deriv_quad8
from pyfem.materials.yield_criterion import StressState2D



#Esta clase tiene que variar para problemas axisimetricos ya que B es una matriz de (4,8)
class Quad8(AreaElement):
    def __init__(self, mater, section, coord, conec, dof): # O que reciba un Gauss_Legendre o un Node4Shape para evitar instancias repetidas
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
    
    def get_body_load(self):
        bforc = np.array([0, -self.mater.dense])
        N_b = bforc @ self.nmatx
        return np.sum(N_b * self.dvolu[:,None], axis=0)
    
    def get_stiff_mat(self):
        bmatx_T = np.transpose(self.bmatx,(0,2,1)) #Transponer B por cada punto de integracion
        BT_D_B = bmatx_T @ self.dmatx @ self.bmatx
        return np.sum(BT_D_B * self.dvolu[:,None,None], axis=0)
    

    def init_element(self):
        points, weights = gauss_nd(2,2)
        npoin = points.shape[0]
        #self.stress = np.zeros((npoin,3))
        self.stress = StressState2D()
        self.yielded = np.zeros(npoin, dtype=bool)
        self.dmatx = np.tile(self.mater.dmatx, (npoin,1,1)) # D = [(3,3),(3,3),(3,3),(3,3)]
        self.bmatx = np.zeros((npoin,3,16)) # B = [(3,16),(3,16),(3,16),(3,16)]
        self.nmatx = np.zeros((npoin,2,16)) # N = [(2,16),(2,16),(2,16),(2,16)]
        det_J = np.zeros(npoin)

        for i, point in enumerate(points):
            self.nmatx[i] = self.get_shape_mat(*point)
            self.bmatx[i], det_J[i] = self.get_strain_mat(*point)
        
        self.area = np.dot(det_J, weights)
        self.dvolu = self.section.thick * det_J * weights
        self.stiff = self.get_stiff_mat()
        self.bload = self.get_body_load()

    def calc_stress(self, glob_disps):
        #Estas partes no importa de momento
        #poins = 1/self.quad_scheme.points
        #gvals = self.shape.funcs(*poins.T).T #4x4
        #order = [0,2,3,1]
        gauss_stress = self.dmatx @ self.bmatx @ glob_disps #4x3
        #nodes_stress = gvals[order] @ gauss_stress[order] #4x3
        return gauss_stress#, nodes_stress
    
    '''
    def update_stiff(self, delta_stress):
        self.stress += delta_stress
        modi_stress = self.const_model.all_components(self.stress)
        prev_yielded = np.copy(self.yielded)
        
        for i, stress in enumerate(modi_stress):
            self.yield_crite.enter_stress(stress)
            
            if self.yield_crite.check_yield():
                if not self.yielded[i]:
                    avect = self.yield_crite.get_flow_vector()
                    self.dmatx[i] = self.mater.calculate_Dp(avect)
                    self.yielded[i] = True

        if not np.array_equal(prev_yielded, self.yielded):
            self.stiff = self.get_stiff_mat()
    '''