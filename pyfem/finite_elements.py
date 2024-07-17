
import numpy as np
import scipy as sp
from scipy import linalg
from .gauss_quad import Gauss_Legendre


def elast_mat_2d(mater):
    E, nu, _ = mater #Elasticity modulus, poisson modulus
    enu = E / (1-nu**2)
    mnu = (1-nu)/2
    D = enu*np.array([[1, nu, 0],
                      [nu, 1, 0],
                      [0, 0, mnu]])
    return D


class FElement():
    def __init__(self, nodes, coord, param, mater): 
        self.coord = coord
        self.nodes = nodes
        self.mater = mater
        self.param = param
        self.nnods = nodes.shape[0]
        self.dof = None

    def set_dof(self, ndofn: int) -> None:
        self.ndofn = ndofn
        node_dof = np.arange(ndofn, dtype=int)
        self.dof = np.repeat(ndofn*self.nodes, ndofn) + np.tile(node_dof, self.nnods)

    

class Quad4(FElement):
    def __init__(self, nodes, coords, params, mater):
        super().__init__(nodes, coords, params, mater)
        self.set_dof(2)
        self.quad_scheme = Gauss_Legendre(2, ndim=2)
        self.thick = self.param
        self.elast = elast_mat_2d(self.mater)
        self.get_stiff_mat()
        self.cracked = False
        
    def shape_funcs(self, r, s):
        N = 0.25*np.array([(1 - r)*(1 - s),
                           (1 - r)*(1 + s), #(1 + r)*(1 - s)
                           (1 + r)*(1 + s), #
                           (1 + r)*(1 - s)]) #(1 - r)*(1 + s)
        dN = 0.25*np.array([[s - 1, -s - 1, s + 1, -s + 1], #[s - 1, -s + 1, s + 1, -s - 1]
                            [r - 1, -r + 1, r + 1, -r - 1]]) #[r - 1, -r - 1, r + 1, -r + 1]
        return N, dN
    
    def get_diffs_mat(self, r, s):
        N, dNdn = self.shape_funcs(r, s)
        J = dNdn @ self.coord
        dNdc = sp.linalg.inv(J) @ dNdn
        H = np.zeros((2, 2*N.shape[0]))
        B = np.zeros((3, 2*N.shape[0]))
        H[0, 0::2] = N
        H[1, 1::2] = N
        B[0, 0::2] = dNdc[0, :]
        B[1, 1::2] = dNdc[1, :]
        B[2, 0::2] = dNdc[1, :]
        B[2, 1::2] = dNdc[0, :]
        return H, B, linalg.det(J)
    
    #En caso de que se necesite optimizar
    def get_strain_mat(self, r, s):
        _, dNdn = self.shape_funcs(r, s)
        J = dNdn @ self.coord
        dNdc = sp.linalg.inv(J) @ dNdn
        B = np.zeros((3, 8))
        B[0, 0::2] = dNdc[0, :]
        B[1, 1::2] = dNdc[1, :]
        B[2, 0::2] = dNdc[1, :]
        B[2, 1::2] = dNdc[0, :]
        return B

    def get_stiff_mat(self):
        points = self.quad_scheme.points
        weights = self.quad_scheme.weights
        t, dens = self.thick, self.mater[-1]
        D = self.elast
        b = np.array([0, dens])
        k_vals = np.zeros((len(points),8,8))
        m_vals = np.zeros((len(points),8,8))
        f_vals = np.zeros((len(points),8))
        for i, point in enumerate(points):
            H, B, detJ = self.get_diffs_mat(*point)
            k_vals[i] = t*(B.T @ D @ B)*detJ
            m_vals[i] = dens*(H.T @ H)*detJ
            f_vals[i] = t*(b @ H)*detJ
            
        self.stiff = np.sum(k_vals * weights[:, None, None], axis=0)
        self.mass = np.sum(m_vals * weights[:, None, None], axis=0)
        self.self_weight = np.sum(f_vals * weights[:, None], axis=0)
    
    def get_stress(self, u):
        points = self.integration_scheme.points
        D = self.elast
        self.stress = np.zeros((3,4))
        gauss_stress = np.zeros((3,4))
        extrapol = np.zeros((4,4)) #extrapolation
        order = [0,3,1,2]
        for i, point in enumerate(points):
            new_point = 1/point
            B = self.get_strain_mat(*point)
            gauss_stress[:, order[i]] = D @ B @ u
            extrapol[:, order[i]] = self.shape_funcs(*new_point)[0]
            self.stress += gauss_stress @ extrapol




class Bar2D(FElement):
    def __init__(self, nodes, coord, param, mater):
        super().__init__(nodes, coord, param, mater)
        self.set_dof(2)
        vector = self.coord[1] - self.coord[0]
        self.xarea = self.param
        self.elast = self.mater[0]
        self.length = sp.linalg.norm(vector)
        self.dirvec = vector/self.length
        self.get_stiff_mat()
        self.get_mass_mat()
    
    def rotation_matrix(self):
        c, s = self.dirvec
        R = np.array([[c, s, 0, 0], 
                      [0, 0, c, s]])
        return R
    
    def get_stiff_mat(self):
        EA_L = self.elast * self.xarea / self.length
        K = EA_L * np.array([[ 1, -1],
                             [-1,  1]])
        R = self.rotation_matrix()
        self.stiff = R.T @ K @ R
    
    def get_mass_mat(self):
        dens = self.mater[-1]
        m = self.xarea * self.length * dens
        self.mass = m/6 * np.array([[2, 0, 1, 0],
                                    [0, 2, 0, 1],
                                    [1, 0, 2, 0],
                                    [0, 1, 0, 2]])
        self.self_weight = np.array([0, m/2, 0, m/2])

    def get_stress(self, u):
        E_L = self.elast / self.length
        c, s = self.dirvec
        B = np.array([-c, -s, c, s])
        self.stress = E_L * B @ u # E*B*u
        self.iforce = self.stress * self.xarea
    

    

class Bar1D(FElement):
    def __init__(self, nodes, coord, param, mater): #mater= [E, nu, sy, Hp, dens]
        super().__init__(nodes, coord, param, mater)
        self.set_dof(1)
        self.xarea = self.param
        self.elast = self.mater[0]
        self.length = np.abs(self.coord[1] - self.coord[0])
        self.get_stiff_mat()
        self.stress = 0.0
        self.yielded = False
    
    def get_stiff_mat(self):
        EA_L = self.elast * self.xarea / self.length
        K = EA_L * np.array([[ 1, -1],
                             [-1,  1]])
        self.stiff = K

    def mod_stiff_mat(self):
        E = self.elast
        Hp = self.mater[3]
        fact = (1-E/(E+Hp))
        self.elast = fact * self.elast
        self.stiff = fact * self.stiff

    def get_stress(self, u):
        sy = self.mater[2]
        E_L = self.elast / self.length
        B = 1/self.length * np.array([-1, 1])
        self.stress += E_L * B @ u
        if abs(self.stress) > sy:
            if not self.yielded:
                self.mod_stiff_mat()
                self.yielded = True
    
    
    

#class Beam2D()