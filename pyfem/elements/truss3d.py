
import numpy as np
import scipy as sp
from .base_elem import Element

class Truss3D(Element):
    def __init__(self, nodes, coord, section, mater): #elast, xarea, i_mom)
        super().__init__(nodes, coord, section, mater)
        self.set_dof(3)
        vector = self.coord[1] - self.coord[0]
        self.xarea = section.xarea
        self.elast = mater.elast
        self.length = sp.linalg.norm(vector)
        self.dirvec = vector/self.length
        self.init_element()

    def init_element(self):
        cx, cy, cz = self.dirvec
        cx2 = cx*cx
        cy2 = cy*cy
        cz2 = cz*cz
        cxy = cx*cy
        cxz = cx*cz
        cyz = cy*cz
        EA_L = self.elast * self.xarea / self.length
        self.stiff = EA_L * np.array([[cx2, cxy, cxz, -cx2, -cxy, -cxz],
                                      [cxy, cy2, cyz, -cxy, -cy2, -cyz],
                                      [cxz, cyz, cz2, -cxz, -cyz, -cz2],
                                      [-cx2, -cxy, -cxz, cx2, cxy, cxz],
                                      [-cxy, -cy2, -cyz, cxy, cy2, cyz],
                                      [-cxz, -cyz, -cz2, cxz, cyz, cz2]])
        
    def calculate_forces(self, glob_disps):
        EA_L = self.elast * self.xarea / self.length
        cx, cy, cz = self.dirvec
        B = np.array([-cx, -cy, -cz, cx, cy, cz])
        self.force = EA_L * B @ glob_disps