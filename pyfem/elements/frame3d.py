
import numpy as np
import scipy as sp
from .base_elem import FrameElement

class Frame3D(FrameElement):
    def __init__(self, mater, section, coord, conec, dof):
        super().__init__(mater, section, coord, conec, dof)
        self.init_element()
    
    def get_local_axes(self):
        e1 = self.coord[1] - self.coord[0] # vector rojo
        e1 = e1 / self.length

        # vector "ref" arriba por defecto; si es colineal con ex se elige alternativa
        ref = np.array([0.0, 0.0, 1.0])
        e3 = ref - e1[2] * e1
        e3_norm = np.linalg.norm(e3)
        if e3_norm <= 1e-8:
            ref = np.array([0.0, 1.0, 0.0])
            e3 = ref - e1[1] * e1
            e3_norm = np.linalg.norm(e3)
            if e3_norm <= 1e-8:
                ref = np.array([1.0, 0.0, 0.0])
                e3 = ref - e1[0] * e1
                e3_norm = np.linalg.norm(e3)

        e3 = e3 / e3_norm

        e2 = np.cross(e3, e1)
        e2 = e2 / np.linalg.norm(e2)

        local_axes = np.vstack((e1, e2, e3))

        # esta parte que se girando la seccion en un angulo arbitrario
        if self.section.section_direc == 90:
            local_axes = np.vstack((e1, e3, -e2))
        elif self.section.section_direc == 0:
            local_axes = np.vstack((e1, e2, e3))
        else:
            raise ValueError("Angle for section orientation must be 90 or 0")
        return local_axes

    def rmatx(self):
        # hacer la prueba
        #rmatx = self.local_axes
        rmatx = self.local_axes.T # vectores como columnas
        R = np.zeros((12,12))
        R[0:3,0:3] = rmatx
        R[3:6,3:6] = rmatx
        R[6:9,6:9] = rmatx
        R[9:12,9:12] = rmatx

        return R

    def kmatx(self):
        EA = self.mater.elast * self.section.xarea
        EIz = self.mater.elast * self.section.inrt3
        EIy = self.mater.elast * self.section.inrt2
        G = self.mater.elast / (2 * (1 + self.mater.poiss))
        oneEA = EA / self.length
        oneGJ = G * self.section.torsion_const / self.length

        twoEIz = 2 * EIz / self.length
        fourEIz = 4 * EIz / self.length
        sixEIz = 6 * EIz / self.length**2
        twelveEIz = 12 * EIz / self.length**3
        
        twoEIy = 2 * EIy / self.length
        fourEIy = 4 * EIy / self.length
        sixEIy = 6 * EIy / self.length**2
        twelveEIy = 12 * EIy / self.length**3
        

        K = np.zeros((12,12))
        K[0, [0,6]] = [ oneEA, -oneEA]
        K[1, [1,5,7,11]] = [ twelveEIz,   sixEIz,   -twelveEIz,   sixEIz]
        K[2, [2,4,8,10]] = [ twelveEIy,  -sixEIy,   -twelveEIy,  -sixEIy]
        K[3, [3,9]] = [ oneGJ, -oneGJ]
        K[4, [2,4,8,10]] = [ -sixEIy,   fourEIy,   sixEIy,   twoEIy]
        K[5, [1,5,7,11]] = [  sixEIz,   fourEIz,  -sixEIz,   twoEIz]
        K[6, [0,6]] = [-oneEA,  oneEA]
        K[7, [1,5,7,11]] = [ -twelveEIz,  -sixEIz,   twelveEIz,  -sixEIz]
        K[8, [2,4,8,10]] = [ -twelveEIy,   sixEIy,   twelveEIy,   sixEIy]
        K[9, [3,9]] = [-oneGJ,  oneGJ]
        K[10, [2,4,8,10]] = [ -sixEIy,   twoEIy,   sixEIy,   fourEIy]
        K[11, [1,5,7,11]] = [  sixEIz,   twoEIz,  -sixEIz,   fourEIz]
        
        return K

    
    def init_element(self):
        self.length = np.linalg.norm(self.coord[1] - self.coord[0])
        self.mid_point = np.mean(self.coord, axis=0)
        self.local_axes = self.get_local_axes()
        
        self.stiff = self.rmatx().T @ self.kmatx() @ self.rmatx()
        #self.mass = self.rmatx().T @ self.mmatx() @ self.rmatx()
        self.loads = np.zeros(12)
        self.force = np.zeros(12)

    '''
    def add_loadss(self, q1i, q2i, q1j, q2j):
        # Añadir en coordenadas locales
        # q1i = intensidad en el nodo i en direccion de la barra
        # q1j = intensidad en el nodo j en direccion de la barra
        # q2i = intensidad en el nodo i en direccion perpendicular de la barra
        # q2j = intensidad en el nodo j en direccion perpendicular de la barra
        qi_qj = q2i - q2j

        self.loads[0] =  (q1i/3 + q1j/6) * self.length
        self.loads[1] = -(3/20 * (qi_qj) - q2i/2) * self.length
        self.loads[2]  = -(1/30 * (qi_qj) - q2i/12) * self.length**2

        self.loads[3] =  (q1j/3 + q1i/6) * self.length
        self.loads[4] = -(7/20 * (qi_qj) - q2i/2) * self.length
        self.loads[5]  =  (1/20 * (qi_qj) - q2i/12) * self.length**2

        self.loads = self.rmatx() @ self.loads
    '''
    
    def calculate_forces(self, glob_disps):
        # A Coordenadas locales
        #self.force = self.rmatx() @ self.stiff @ glob_disps
        self.force = self.rmatx().T @ self.stiff @ glob_disps