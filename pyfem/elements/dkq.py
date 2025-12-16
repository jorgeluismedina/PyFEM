
import numpy as np
import scipy as sp
from pyfem.elements.base_elem import AreaElement
from pyfem.gauss_quad import gauss_nd
from pyfem.shape_funcs import shape_quad4, deriv_quad8, deriv_quad4
from pyfem.materials.yield_criterion import StressState2D



class DKQ(AreaElement):
    def __init__(self, mater, section, coord, conec, dof):
        super().__init__(mater, section, coord, conec, dof)
        #self.yield_crite = self.mater.yield_crite
        self.init_element()
       
    def get_shape_mat(self, r, s):
        shape = shape_quad4(r,s)
        N = np.zeros((3,12)) #(3,3*nnods)
        N[0, 0::3] = shape
        N[1, 1::3] = shape
        N[2, 2::3] = shape
        return N 
    

    def get_const_mat(self):
        t = self.section.thick
        E = self.mater.elast
        nu = self.mater.poiss
        kf = E * t**3 / (12 * (1 - nu**2))

        Df = np.zeros((3,3))
        Df[0,0] = kf
        Df[0,1] = kf * nu
        Df[1,0] = kf * nu
        Df[1,1] = kf
        Df[2,2] = kf * (1 - nu) / 2

        return Df
    

    def get_interpolation_mats(self, r, s):
        side_vectors = np.diff(self.coord, axis=0, append=self.coord[0,None])
        xij = side_vectors[:,0]
        yij = side_vectors[:,1]
        #x = self.coord[:,0]
        #y = self.coord[:,1]
        #xij = np.array([x[0]-x[1], x[1]-x[2], x[2]-x[3], x[3]-x[0]])
        #yij = np.array([y[0]-y[1], y[1]-y[2], y[2]-y[3], y[3]-y[0]])
        Lij2 = xij**2 + yij**2

        a = -xij / Lij2 # a5, a6, a7, a8
        b = 0.75 * xij * yij / Lij2 # b5, b6, b7, b8
        c = (0.25*xij**2 - 0.5*yij**2) / Lij2
        d = -yij / Lij2
        e = (0.25*yij**2 - 0.5*xij**2) / Lij2

        deriv = deriv_quad8(r, s)
        dNco = deriv[:,0::2] # corner nodes 1,2,3,4 derivatives
        dNsi = deriv[:,1::2] # sides nodes 5,6,7,8 derivatives

        adN = a * dNsi
        bdN = b * dNsi
        cdN = c * dNsi
        ddN = d * dNsi
        edN = e * dNsi

        adN_m_adN = adN - np.roll(adN, 1, axis=1)
        bdN_p_bdN = bdN + np.roll(bdN, 1, axis=1)
        cdN_p_cdN = cdN + np.roll(cdN, 1, axis=1)
        ddN_m_ddN = ddN - np.roll(ddN, 1, axis=1)
        edN_p_edN = edN + np.roll(edN, 1, axis=1)

        dHx = np.empty((2,12))
        dHx[:,0::3] = 1.5 * adN_m_adN
        dHx[:,1::3] = bdN_p_bdN
        dHx[:,2::3] = dNco - cdN_p_cdN

        dHy = np.empty((2,12))
        dHy[:,0::3] = 1.5 * ddN_m_ddN
        dHy[:,1::3] = -dNco + edN_p_edN
        dHy[:,2::3] = -bdN_p_bdN

        return dHx, dHy
    
    def get_strain_mat(self, r, s):
        derivHx, derivHy = self.get_interpolation_mats(r, s)
        jacob = deriv_quad4(r, s) @ self.coord
        inv_J = sp.linalg.inv(jacob)
        cartHx = inv_J @ derivHx
        cartHy = inv_J @ derivHy
        B = np.zeros((3,12))
        B[0] = cartHx[0] #dHxdx
        B[1] = cartHy[1] #dHydy
        B[2] = cartHx[1] + cartHy[0] #dHxdy + dHydx 

        return B, sp.linalg.det(jacob)

    
    def get_stiff_mat(self):
        # Se debe integrar sobre el area, por que el espesor 
        # ya esta incluido en las matrices Ds y Ds
        bmatx_T = np.transpose(self.bmatx,(0,2,1)) #Transponer B por cada punto de integracion
        BT_D_B = bmatx_T @ self.dmatx @ self.bmatx
        K = np.sum(BT_D_B * self.darea[:,None,None], axis=0)
        return K
    

    def init_element(self):
        points, weights = gauss_nd(2,2)
        npoin = points.shape[0]
        #self.stress = np.zeros((npoin,3))
        #self.stress = StressState2D()
        #self.yielded = np.zeros(npoin, dtype=bool)
        self.dmatx = np.tile(self.get_const_mat(), (npoin,1,1)) # D = [(3,3),(3,3),(3,3),(3,3)]
        self.bmatx = np.zeros((npoin,3,12)) # B = [(3,16),(3,16),(3,16),(3,16)]
        #self.nmatx = np.zeros((npoin,2,16)) # N = [(2,16),(2,16),(2,16),(2,16)]
        det_J = np.zeros(npoin)

        for i, point in enumerate(points):
            #self.nmatx[i] = self.get_shape_mat(*point)
            self.bmatx[i], det_J[i] = self.get_strain_mat(*point)
        
        self.area = np.dot(det_J, weights)
        self.darea = det_J * weights
        self.stiff = self.get_stiff_mat()
        self.loads = np.zeros(12)
        self.add_selfweight()

    def add_selfweight(self):
        Nb_nodal = -self.mater.dense * self.area * self.section.thick / 4 # conversion a carga de area
        self.loads[0::3] += Nb_nodal
    
    def add_loads(self, q_normal): # carga en direccion z global
        Nq_nodal = self.area * q_normal / 4
        self.loads[0::3] += Nq_nodal

    def calculate_stress(self, glob_disps):
        #Estas partes no importa de momento
        #poins = 1/self.quad_scheme.points
        #gvals = self.shape.funcs(*poins.T).T #4x4
        #order = [0,2,3,1]
        #siguen habiendo 4 puntos de integracion
        gauss_stress = self.dmatx @ self.bmatx @ glob_disps #4x3
        #nodes_stress = gvals[order] @ gauss_stress[order] #4x3
        #self.stress.update(gauss_stress[:,0], 
        #                   gauss_stress[:,1], 
        #                   gauss_stress[:,2])