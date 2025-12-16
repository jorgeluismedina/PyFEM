
import numpy as np
import scipy as sp
from pyfem.elements.base_elem import AreaElement
from pyfem.gauss_quad import gauss_nd
from pyfem.shape_funcs import shape_quad4, deriv_quad4
from pyfem.materials.yield_criterion import StressState2D



#Esta clase tiene que variar para problemas axisimetricos ya que B es una matriz de (4,8)
class Quad4(AreaElement):
    def __init__(self, mater, section, coord, conec, dof): # O que reciba un Gauss_Legendre o un Node4Shape para evitar instancias repetidas
        super().__init__(mater, section, coord, conec, dof)
        #self.yield_crite = self.mater.yield_crite
        self.init_element()
    
    
    def get_shape_mat(self, r, s):
        shape = shape_quad4(r,s)
        N = np.zeros((2,8)) #(2,2*nnods)
        N[0, 0::2] = shape
        N[1, 1::2] = shape
        return N 
    
    def get_strain_mat(self, r, s):
        deriv = deriv_quad4(r,s)
        jacob = deriv @ self.coord
        cartd = sp.linalg.inv(jacob) @ deriv
        B = np.zeros((3,8)) #(3,2*nnods)
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
        self.bmatx = np.zeros((npoin,3,8)) # B = [(3,8),(3,8),(3,8),(3,8)]
        self.nmatx = np.zeros((npoin,2,8)) # N = [(2,8),(2,8),(2,8),(2,8)]
        det_J = np.zeros(npoin)

        for i, point in enumerate(points):
            self.nmatx[i] = self.get_shape_mat(*point)
            self.bmatx[i], det_J[i] = self.get_strain_mat(*point)
        
        self.area = np.dot(det_J, weights)
        self.darea = det_J * weights
        self.stiff = self.get_stiff_mat()
        #self.bload = self.get_body_load()
        self.loads = np.zeros(8)
        self.add_selfweight()

    def add_selfweight(self):
        Nb_nodal = -self.mater.dense * self.section.thick * self.area / 4
        self.loads[1::2] += Nb_nodal
        

    # normal_load es positivo si es como succion y negativo como presion
    # tangent_load es positivo en sentido antihorario y negativo en sentido horario
    # el primer lado corresponde a los dos primeros nodos
    # solo sirve para carga distribuida lineal
    def add_loads(self, normal_distributed=[0,0,0,0], tangent_distributed=[0,0,0,0]):
        side_vectors = np.diff(self.coord, axis=0, append=self.coord[0,None])
        side_lenghts = np.linalg.norm(side_vectors, axis=1)
        c, s = (side_vectors / side_lenghts[:, None]).T
        n_equi = np.array(normal_distributed) * side_lenghts / 2
        t_equi = np.array(tangent_distributed) * side_lenghts / 2

        #los cosenos negativos porque en antihorario el x positivo apunta a la izquierda
        x_equi =  -t_equi*c - n_equi*s 
        y_equi =   t_equi*s - n_equi*c
         
        self.loads[0::2] += x_equi + np.roll(x_equi,1)
        self.loads[1::2] += y_equi + np.roll(y_equi,1)
        #self.loads = side_lenghts


    def calculate_stress(self, glob_disps):
        gauss_stress = self.dmatx @ self.bmatx @ glob_disps
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
