
import numpy as np
import scipy as sp
from pyfem.elements.base_elem import Element
from pyfem.gauss_quad import gauss_nd
from pyfem.shape_funcs import shape_hex8, deriv_hex8
from pyfem.materials.yield_criterion import StressState2D



#Esta clase tiene que variar para problemas axisimetricos ya que B es una matriz de (4,8)
class Hex8(Element):
    def __init__(self, mater, coord, conec, dof): # O que reciba un Gauss_Legendre o un Node4Shape para evitar instancias repetidas
        super().__init__(mater, coord, conec, dof)
        #self.yield_crite = self.mater.yield_crite
        self.init_element()
    
    def get_shape_mat(self, r, s, t):
        shape = shape_hex8(r,s,t)
        N = np.zeros((3,24)) #(3,2*nnods)
        N[0, 0::3] = shape
        N[1, 1::3] = shape
        N[2, 2::3] = shape
        return N 
    
    def get_strain_mat(self, r, s, t):
        deriv = deriv_hex8(r,s,t)
        jacob = deriv @ self.coord
        cartd = sp.linalg.inv(jacob) @ deriv
        B = np.zeros((6,24)) #(6,3*nnods)
        B[0, 0::3] = cartd[0, :]
        B[1, 1::3] = cartd[1, :]
        B[2, 2::3] = cartd[2, :]
        B[3, 0::3] = cartd[1, :]
        B[3, 1::3] = cartd[0, :]
        B[4, 0::3] = cartd[2, :]
        B[4, 2::3] = cartd[0, :]
        B[5, 1::3] = cartd[2, :]
        B[5, 2::3] = cartd[1, :]
        return B, sp.linalg.det(jacob)
    
    def get_body_load(self):
        bforc = np.array([0, 0, -self.mater.dense])
        N_b = bforc @ self.nmatx
        return np.sum(N_b * self.dvolu[:,None], axis=0)
    
    def get_stiff_mat(self):
        bmatx_T = np.transpose(self.bmatx,(0,2,1)) #Transponer B por cada punto de integracion
        BT_D_B = bmatx_T @ self.dmatx @ self.bmatx
        return np.sum(BT_D_B * self.dvolu[:,None,None], axis=0)
    

    def init_element(self):
        points, weights = gauss_nd(2,3)
        npoin = points.shape[0]
        #self.stress = np.zeros((npoin,3))
        #self.stress = StressState2D()
        self.yielded = np.zeros(npoin, dtype=bool)
        self.dmatx = np.tile(self.mater.dmatx, (npoin,1,1)) # D = [(3,3),(3,3),(3,3),(3,3)]
        self.bmatx = np.zeros((npoin,6,24)) # B = [(3,8),(3,8),(3,8),(3,8)]
        self.nmatx = np.zeros((npoin,3,24)) # N = [(2,8),(2,8),(2,8),(2,8)]
        det_J = np.zeros(npoin)

        for i, point in enumerate(points):
            self.nmatx[i] = self.get_shape_mat(*point)
            self.bmatx[i], det_J[i] = self.get_strain_mat(*point)
        
        self.volume = np.dot(det_J, weights)
        self.dvolu = det_J * weights
        self.stiff = self.get_stiff_mat()
        self.bload = self.get_body_load()
        self.loads = np.zeros(16)

    # modificar para hex8
    # normal_load es positivo si es como succion y negativo como presion
    # tangent_load es positivo en sentido antihorario y negativo en sentido horario
    def add_surface_loads(self, normal_distributed=[0,0,0,0,0,0], tangent_distributed=[0,0,0,0,0,0]):
        side_vectors = np.diff(self.coord, axis=0, append=self.coord[0,None])
        side_lenghts = np.linalg.norm(side_vectors, axis=1)
        c, s = (side_vectors / side_lenghts[:, None]).T
        n_equi = np.array(normal_distributed) * side_lenghts / 2
        t_equi = np.array(tangent_distributed) * side_lenghts / 2

        #print(n_equi)
        #print(t_equi)
        #print('\n')

        x_equi = -n_equi*c - t_equi*s #todo menos por que antihorario es positivo
        y_equi = -n_equi*s + t_equi*c
         
        self.loads[0::2] = x_equi + np.roll(x_equi,1)
        self.loads[1::2] = y_equi + np.roll(y_equi,1)
        #self.loads = side_lenghts

    def calculate_stress(self, glob_disps):
        #Estas partes no importa de momento
        #poins = 1/self.quad_scheme.points
        #gvals = self.shape.funcs(*poins.T).T #4x4
        #order = [0,2,3,1]
        gauss_stress = self.dmatx @ self.bmatx @ glob_disps #4x3
        #nodes_stress = gvals[order] @ gauss_stress[order] #4x3
        #self.gstress = gauss_stress#, nodes_stress
        self.stress = gauss_stress #cambiar luego
        #return gauss_stress
    '''
        self.stress.update(gauss_stress[:,0], 
                           gauss_stress[:,1], 
                           gauss_stress[:,2])

    '''