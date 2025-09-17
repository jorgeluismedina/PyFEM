
import numpy as np
import scipy as sp
from pyfem.elements.base_elem import Element
from pyfem.gauss_quad import Gauss_Legendre
from pyfem.shape_funcs import Node4ShapeA, Node4ShapeH
from pyfem.materials.yield_criterion import StressState2D



#Esta clase tiene que variar para problemas axisimetricos ya que B es una matriz de (4,8)
class Quad4(Element):
    def __init__(self, conec, dof, coord, section, mater): # O que reciba un Gauss_Legendre o un Node4Shape para evitar instancias repetidas
        super().__init__(conec, dof, coord, section, mater)
        #self.yield_crite = self.mater.yield_crite
        #self.const_model = self.mater.const_model
        self.shape = Node4ShapeH() # Que esta clase sean funciones de esta misma clase para ahorrar memoria (instancias repetidas)
        self.quad_scheme = Gauss_Legendre(2, ndim=2) # Que Gauss_Legendre solo sea una funcion para ahorrar (instancias repetidas)
        self.init_element()
    
    
    def get_shape_mat(self, r, s):
        shape = self.shape.funcs(r,s)
        N = np.zeros((2,8)) #(2,2*nnods)
        N[0, 0::2] = shape
        N[1, 1::2] = shape
        return N 
    
    def get_strain_mat(self, r, s):
        deriv = self.shape.deriv(r,s)
        jacob = deriv @ self.coord
        cartd = sp.linalg.inv(jacob) @ deriv
        B = np.zeros((3,8)) #(3,2*nnods)
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
        npoin = self.quad_scheme.npoin
        #self.stress = np.zeros((npoin,3))
        self.stress = StressState2D()
        self.yielded = np.zeros(npoin, dtype=bool)
        # modificar
        const = self.mater.elast/(1-self.mater.poiss**2)
        conss = (1-self.mater.poiss)/2
        dmatx = const*np.array([[1, self.mater.poiss, 0],
                                [self.mater.poiss, 1, 0],
                                [0, 0, conss]])
        self.dmatx = np.tile(dmatx, (npoin,1,1))
        #-------------------------------------------
        self.bmatx = np.zeros((npoin,3,8)) # B = [(3,8),(3,8),(3,8),(3,8)]
        self.nmatx = np.zeros((npoin,2,8)) # N = [(2,8),(2,8),(2,8),(2,8)]
        det_J = np.zeros(npoin)

        for i, point in enumerate(self.quad_scheme.points):
            self.nmatx[i] = self.get_shape_mat(*point)
            self.bmatx[i], det_J[i] = self.get_strain_mat(*point)
        
        self.area = np.dot(det_J, self.quad_scheme.weights)
        self.dvolu = self.section.thick * det_J * self.quad_scheme.weights
        self.stiff = self.get_stiff_mat()
        self.bload = self.get_body_load()

    def calculate_stress(self, glob_disps):
        #Estas partes no importa de momento
        #poins = 1/self.quad_scheme.points
        #gvals = self.shape.funcs(*poins.T).T #4x4
        #order = [0,2,3,1]
        gauss_stress = self.dmatx @ self.bmatx @ glob_disps #4x3
        #nodes_stress = gvals[order] @ gauss_stress[order] #4x3
        #self.gstress = gauss_stress#, nodes_stress
        #self.stress = gauss_stress #cambiar luego
        self.stress.update(gauss_stress[:,0], 
                           gauss_stress[:,1], 
                           gauss_stress[:,2])
    

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
