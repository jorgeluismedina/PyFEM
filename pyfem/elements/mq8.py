
import numpy as np
import scipy as sp
from pyfem.elements.base_elem import AreaElement
from pyfem.gauss_quad import gauss_nd
from pyfem.shape_funcs import shape_quad8, deriv_quad8
from pyfem.materials.yield_criterion import StressState2D



class MQ8(AreaElement):
    def __init__(self, mater, section, coord, conec, dof): # O que reciba un Gauss_Legendre o un Node4Shape para evitar instancias repetidas
        super().__init__(mater, section, coord, conec, dof)
        #self.yield_crite = self.mater.yield_crite
        self.init_element()
       
    def get_shape_mat(self, r, s):
        shape = shape_quad8(r,s)
        N = np.zeros((3,24)) #(3,3*nnods)
        N[0, 0::3] = shape
        N[1, 1::3] = shape
        N[2, 2::3] = shape
        return N 
    
    def get_strain_mats(self, r, s): #Bf and Bs
        shape = shape_quad8(r,s)
        deriv = deriv_quad8(r,s)
        jacob = deriv @ self.coord
        cartd = sp.linalg.inv(jacob) @ deriv

        Bf = np.zeros((3,24)) #(3,3*nnods)
        Bf[0, 1::3] = -cartd[0, :]
        Bf[1, 2::3] = -cartd[1, :]
        Bf[2, 1::3] = -cartd[1, :]
        Bf[2, 2::3] = -cartd[0, :]

        Bs = np.zeros((2,24)) #(2,3*nnods)
        Bs[0, 0::3] =  cartd[0, :]
        Bs[1, 0::3] =  cartd[1, :]
        Bs[0, 1::3] = -shape
        Bs[1, 2::3] = -shape
        return Bf, Bs, sp.linalg.det(jacob)

    
    def get_const_mats(self):
        t = self.section.thick
        E = self.mater.elast
        nu = self.mater.poiss
        kf = E * t**3 / (12 * (1 - nu**2))
        ks = E * t / (2.4 * (1 + nu))

        Df = np.zeros((3,3))
        Df[0,0] = kf
        Df[0,1] = kf * nu
        Df[1,0] = kf * nu
        Df[1,1] = kf
        Df[2,2] = kf * (1 - nu) / 2

        Ds = np.zeros((2,2))
        Ds[0,0] = ks
        Ds[1,1] = ks
        return Df, Ds


    #'''
    def get_stiff_mat(self, bmatx, dmatx):
        # Se debe integrar sobre el area, por que el espesor 
        # ya esta incluido en las matrices Ds y Ds
        bmatx_T = np.transpose(bmatx,(0,2,1)) #Transponer B por cada punto de integracion
        BT_D_B = bmatx_T @ dmatx @ bmatx
        K = np.sum(BT_D_B * self.darea[:,None,None], axis=0)
        return K
    #'''
    

    def init_element(self):
        points, weights = gauss_nd(2,2)
        npoin = points.shape[0]
        #self.stress = np.zeros((npoin,3))
        #self.stress = StressState2D()
        #self.yielded = np.zeros(npoin, dtype=bool)
        Df, Ds = self.get_const_mats()
        self.dfmat = np.tile(Df, (npoin,1,1)) # D = [(3,3),(3,3),(3,3),(3,3)]
        self.dsmat = np.tile(Ds, (npoin,1,1)) # D = [(2,2),(2,2),(2,2),(2,2)]
        self.bfmat = np.zeros((npoin,3,24)) # B = [(3,24),(3,24),(3,24),(3,24)]
        self.bsmat = np.zeros((npoin,2,24)) # B = [(2,24),(2,24),(2,24),(2,24)]
        self.nmatx = np.zeros((npoin,8)) # N = [(8),(8),(8),(8)] Solo existe carga en direccion z global
        det_J = np.zeros(npoin)

        for i, point in enumerate(points):
            self.nmatx[i] = shape_quad8(*point)
            self.bfmat[i], self.bsmat[i], det_J[i] = self.get_strain_mats(*point)
        
        self.area = np.dot(det_J, weights)
        self.darea = det_J * weights
        kfmat = self.get_stiff_mat(self.bfmat, self.dfmat)
        ksmat = self.get_stiff_mat(self.bsmat, self.dsmat)
        self.stiff = kfmat + ksmat
        self.loads = np.zeros(24)
        self.add_selfweight()

    def add_selfweight(self):
        Nb = -self.mater.dense * self.section.thick # conversion a carga de area [N/m2]
        Nb_nodal = np.sum(Nb * self.nmatx * self.darea[:,None], axis=0)
        self.loads[0::3] += Nb_nodal

    def add_loads(self, q_normal): # carga en direccion z global [N/m2]
        Nq = q_normal * self.nmatx #integrando
        Nq_nodal = np.sum(Nq * self.darea[:,None], axis=0)
        self.loads[0::3] += Nq_nodal # el primer dof corresponde al desp en z

    def calculate_stress(self, glob_disps):
        #Estas partes no importa de momento
        #poins = 1/self.quad_scheme.points
        #gvals = self.shape.funcs(*poins.T).T #4x4
        #order = [0,2,3,1]
        #siguen habiendo 4 puntos de integracion
        gauss_fstress = self.dfmat @ self.bfmat @ glob_disps #4x3
        gauss_sstress = self.dsmat @ self.bsmat @ glob_disps #4x2
        #nodes_stress = gvals[order] @ gauss_stress[order] #4x3
        #self.stress.update(gauss_stress[:,0], 
        #                   gauss_stress[:,1], 
        #                   gauss_stress[:,2])
        #return gauss_fstress, gauss_sstress#, nodes_stress
        self.gfstress = gauss_fstress
        self.gsstress = gauss_sstress