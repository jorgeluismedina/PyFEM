
import numpy as np
import scipy as sp
from pyfem.elements.base_elem import AreaElement
from pyfem.gauss_quad import gauss_nd
from pyfem.shape_funcs import shape_quad8, deriv_quad8
from pyfem.materials.yield_criterion import StressState2D

class ShellMQ8(AreaElement):
    def __init__(self, mater, section, coord, conec, dof): # O que reciba un Gauss_Legendre o un Node4Shape para evitar instancias repetidas
        super().__init__(mater, section, coord, conec, dof)
        #self.yield_crite = self.mater.yield_crite
        self.init_element()

    def get_local_axes(self):
        e1 = self.coord[1] - self.coord[0] 
        e1 = e1 / np.linalg.norm(e1) # vector rojo

        aux = self.coord[3] - self.coord[0]
        e3 = np.cross(e1, aux) # perpendicular al plano
        e3 = e3 / np.linalg.norm(e3)
        e2 = np.cross(e3, e1)
        e2 = e2 / np.linalg.norm(e2)

        local_axes = np.vstack((e1, e2, e3))
        return local_axes
    
    # se deberia llamar matriz de transformacion 
    def get_transformation_mat(self):
        #rmatx = self.local_axes
        rmatx = self.local_axes.T # vectores como columnas 
        Tn = np.zeros((6,6))
        Tn[0:3, 0:3] = rmatx
        Tn[3:6, 3:6] = rmatx
        T = np.kron(np.eye(8), Tn)
        return T
    
    def get_local_coords(self):
        rmatx = self.local_axes.T
        loc_xyz = rmatx.T @ (self.coord - self.mid_point).T
        loc_coord = loc_xyz[0:2].T
        return loc_coord

       
    def get_shape_mat(self, r, s):
        shape = shape_quad8(r,s)
        N = np.zeros((6,48)) #(6,6*nnods)
        N[0, 0::6] = shape
        N[1, 1::6] = shape
        N[2, 2::6] = shape
        N[3, 3::6] = shape
        N[4, 4::6] = shape
        N[5, 5::6] = shape
        return N 
    
    def get_strain_mats(self, r, s): #Bf and Bs
        shape = shape_quad8(r,s)
        deriv = deriv_quad8(r,s)
        jacob = deriv @ self.local_coord
        cartd = sp.linalg.inv(jacob) @ deriv

        Ba = np.zeros((3,16)) #(3,2*nnods)
        Ba[0, 0::2] = cartd[0, :]
        Ba[1, 1::2] = cartd[1, :]
        Ba[2, 0::2] = cartd[1, :]
        Ba[2, 1::2] = cartd[0, :]

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
        return Ba, Bf, Bs, sp.linalg.det(jacob)

    
    def get_const_mats(self):
        t = self.section.thick
        E = self.mater.elast
        nu = self.mater.poiss

        ka = E * t / (1 - nu**2)
        kf = E * t**3 / (12 * (1 - nu**2))
        ks = E * t / (2.4 * (1 + nu))

        Da = np.zeros((3,3))
        Da[0,0] = ka
        Da[0,1] = ka * nu
        Da[1,0] = ka * nu
        Da[1,1] = ka
        Da[2,2] = ka * (1 - nu) / 2

        Df = np.zeros((3,3))
        Df[0,0] = kf
        Df[0,1] = kf * nu
        Df[1,0] = kf * nu
        Df[1,1] = kf
        Df[2,2] = kf * (1 - nu) / 2

        Ds = np.zeros((2,2))
        Ds[0,0] = ks
        Ds[1,1] = ks
        return Da, Df, Ds


    #'''
    def assemble_stiff_mat(self, kmemb, kplat):
        Kl = np.zeros((48, 48))
        nnod = 8    
        
        for i, j in np.ndindex(nnod, nnod):
            shi = i * 6 # dof per node shell
            shj = j * 6 

            mei = i * 2 # dof per node membrane
            mej = j * 2

            pli = i * 3 # dof per node plate
            plj = j * 3
            # Membrana (u,v)
            kmembij = kmemb[mei:mei+2, mej:mej+2]
            Kl[shi:shi+2, shj:shj+2] = kmembij
            # Mindlin (w,R_x,R_y)
            kplatij = kplat[pli:pli+3, plj:plj+3]
            Kl[shi+2:shi+5, shj+2:shj+5] = kplatij
            # Drilling stiffnes
            Kl[shi+5, shj+5] = (kplatij[1,1] + kplatij[2,2]) / 2
        
        
        T = self.get_transformation_mat()
        Kg = T @ Kl @ T.T #explacion de esto urgente
        #Kg = T.T @ Kl @ T
        return Kg

    
    def get_stiff_mat(self, bmatx, dmatx):
        # Se debe integrar sobre el area, por que el espesor 
        # ya esta incluido en las matrices Df y Ds
        bmatx_T = np.transpose(bmatx,(0,2,1)) #Transponer B por cada punto de integracion
        BT_D_B = bmatx_T @ dmatx @ bmatx
        K = np.sum(BT_D_B * self.darea[:,None,None], axis=0)
        return K
    #'''
    

    def init_element(self):
        self.mid_point = np.mean(self.coord, axis=0)
        self.local_axes = self.get_local_axes() 
        self.local_coord = self.get_local_coords()

        points, weights = gauss_nd(2,2) #(3,2)
        npoin = points.shape[0]
        #self.stress = np.zeros((npoin,3))
        #self.stress = StressState2D()
        #self.yielded = np.zeros(npoin, dtype=bool)
        Da, Df, Ds = self.get_const_mats()
        self.damat = np.tile(Da, (npoin,1,1)) 
        self.dfmat = np.tile(Df, (npoin,1,1)) 
        self.dsmat = np.tile(Ds, (npoin,1,1)) 
        self.bamat = np.zeros((npoin,3,16))
        self.bfmat = np.zeros((npoin,3,24)) 
        self.bsmat = np.zeros((npoin,2,24)) 
        self.nmatx = np.zeros((npoin,8)) 
        det_J = np.zeros(npoin)
        self.loads = np.zeros(48)

        for i, point in enumerate(points):
            self.nmatx[i] = shape_quad8(*point)
            self.bamat[i], self.bfmat[i], self.bsmat[i], det_J[i] = self.get_strain_mats(*point)
        
        self.area = np.dot(det_J, weights)
        self.darea = det_J * weights
        kmemb = self.get_stiff_mat(self.bamat, self.damat)
        #print(kmemb, '\n')
        kfmat = self.get_stiff_mat(self.bfmat, self.dfmat)
        ksmat = self.get_stiff_mat(self.bsmat, self.dsmat)
        kplat = kfmat + ksmat

        self.stiff = self.assemble_stiff_mat(kmemb, kplat)
        #self.add_selfweight()

    def add_selfweight(self): # Esta fuerza al final debe rotarse para que sea global
        Nb = -self.mater.dense * self.section.thick * self.nmatx # conversion a carga de area [N/m2]
        Nb_nodal = np.sum(Nb * self.darea[:,None], axis=0)
        self.loads[2::6] += Nb_nodal

    def add_loads(self, q_normal): # carga en direccion z global [N/m2] Debe rotarse
        Nq = q_normal * self.nmatx #integrando
        Nq_nodal = np.sum(Nq * self.darea[:,None], axis=0)
        self.loads[2::6] += Nq_nodal # el primer dof corresponde al desp en z


    def calculate_stress(self, glob_disps):
        #siguen habiendo 4 puntos de integracion
        # aca deben convertirse los desplazmientos a local
        gauss_fstress = self.dfmat @ self.bfmat @ glob_disps #4x3
        gauss_sstress = self.dsmat @ self.bsmat @ glob_disps #4x2
        #nodes_stress = gvals[order] @ gauss_stress[order] #4x3
        #self.stress.update(gauss_stress[:,0], 
        #                   gauss_stress[:,1], 
        #                   gauss_stress[:,2])
        #return gauss_fstress, gauss_sstress#, nodes_stress
        self.gfstress = gauss_fstress
        self.gsstress = gauss_sstress