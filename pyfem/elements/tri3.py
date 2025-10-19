

import numpy as np
import scipy as sp
from pyfem.elements.base_elem import AreaElement
from pyfem.materials.yield_criterion import StressState2D



#Esta clase tiene que variar para problemas axisimetricos ya que B es una matriz de (4,8)
class Tri3(AreaElement):
    def __init__(self, mater, section, coord, conec, dof):
        super().__init__(mater, section, coord, conec, dof)
        #self.yield_crite = self.mater.yield_crite
        self.init_element() 
    
    #def get_shape_mat(self, r, s):
    #    shape = self.shape.funcs(r,s)
    #    N = np.zeros((2,8)) #(2,2*nnods)
    #    N[0, 0::2] = shape
    #    N[1, 1::2] = shape
    #    return N 
    
    def get_strain_mat(self):
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
        #self.stress = np.zeros(3)
        self.stress = StressState2D()
        self.yielded = False
        self.dmatx = self.mater.dmatx
        self.bmatx, self.area = self.get_strain_mat() # B = [(3,8),(3,8),(3,8),(3,8)]
        #self.nmatx = np.zeros((npoin,2,8)) # N = [(2,8),(2,8),(2,8),(2,8)]
        self.stiff = self.get_stiff_mat()
        self.bload = self.get_body_load()
        self.loads = np.zeros(6)

    # normal_load es positivo si es como succion y negativo como presion
    # tangent_load es positivo en sentido antihorario y negativo en sentido horario
    def add_loads(self, normal_distributed=[0,0,0], tangent_distributed=[0,0,0]):
        side_vectors = np.array([self.coord[1] - self.coord[0],
                                 self.coord[2] - self.coord[1],
                                 self.coord[0] - self.coord[2]])
        
        side_lenghts = np.linalg.norm(side_vectors, axis=1)
        c, s = (side_vectors / side_lenghts[:, None]).T # cosenos directores
        n_equi = np.array(normal_distributed) * side_lenghts / 2
        t_equi = np.array(tangent_distributed) * side_lenghts / 2

        print(n_equi)
        print(t_equi)
        print('\n')

        x_equi = -n_equi*c - t_equi*s #todo menos por que antihorario es positivo
        y_equi = -n_equi*s + t_equi*c
         
        self.loads[0::2] = x_equi + np.roll(x_equi,1)
        self.loads[1::2] = y_equi + np.roll(y_equi,1)
        #self.loads = side_lenghts



    def calculate_stress(self, glob_disps):
        stress = self.dmatx @ self.bmatx @ glob_disps #4x3
        self.stress.update(*stress)