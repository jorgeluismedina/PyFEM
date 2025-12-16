
import numpy as np


class FrameSection2D:
    def __init__(self, xarea, inrt3):
        self.xarea = xarea
        self.inrt3 = inrt3



# de momenento solo secciones bisimetricas
class FrameSection3D:
    def __init__(self, xarea, inrt3, inrt2, torsion_const, section_direc):
        self.xarea = xarea
        self.inrt3 = inrt3
        self.inrt2 = inrt2
        self.torsion_const = torsion_const
        self.section_direc = section_direc



class RectangularSection:
    def __init__(self, base, height, section_direc):
        self.b = base
        self.h = height
        self.section_direc = section_direc #0 a 90
        self.set_section_props()

    def set_section_props(self):
        b = self.b
        h = self.h
        self.xarea = b*h
        self.inrt3 = b*h**3/12
        self.inrt2 = h*b**3/12
        beta = 1/3 - 0.21 * b/h * (1 - 1/12 * (b/h)**4)
        self.torsion_const = beta * b**3 * h





class AreaSection:
    def __init__(self, thick):
        self.thick = thick