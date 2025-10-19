
import numpy as np
import scipy as sp
from .base_elem import FrameElement

class Truss3D(FrameElement):
    def __init__(self, mater, section, coord, conec, dof): #elast, xarea, i_mom)
        super().__init__(mater, section, coord, conec, dof)
        vector = self.coord[1] - self.coord[0]
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
        
        EA_L = self.mater.elast * self.section.xarea / self.length
        self.stiff = EA_L * np.array([[cx2, cxy, cxz, -cx2, -cxy, -cxz],
                                      [cxy, cy2, cyz, -cxy, -cy2, -cyz],
                                      [cxz, cyz, cz2, -cxz, -cyz, -cz2],
                                      [-cx2, -cxy, -cxz, cx2, cxy, cxz],
                                      [-cxy, -cy2, -cyz, cxy, cy2, cyz],
                                      [-cxz, -cyz, -cz2, cxz, cyz, cz2]])
        
        pAL = self.mater.dense * self.section.xarea * self.length
        self.mass = pAL/6 * np.array([[2*cx2, 2*cxy, 2*cxz, cx2, cxy, cxz],
                                      [2*cxy, 2*cy2, 2*cyz, cxy, cy2, cyz],
                                      [2*cxz, 2*cyz, 2*cz2, cxz, cyz, cz2],
                                      [cx2, cxy, cxz, 2*cx2, 2*cxy, 2*cxz],
                                      [cxy, cy2, cyz, 2*cxy, 2*cy2, 2*cyz],
                                      [cxz, cyz, cz2, 2*cxz, 2*cyz, 2*cz2]])

        
    def calculate_forces(self, glob_disps):
        EA_L = self.mater.elast * self.section.xarea / self.length
        cx, cy, cz = self.dirvec
        B = np.array([-cx, -cy, -cz, cx, cy, cz])
        self.force = EA_L * B @ glob_disps