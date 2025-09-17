
import numpy as np
from abc import ABC, abstractmethod

class Element(ABC):
    def __init__(self, conec, dof, coord, section, mater): # Que no tenga el atributo section
        self.dof = dof
        self.conec = conec
        self.coord = coord
        self.mater = mater
        self.section = section
        self.nnods = conec.shape[0]
        self.loads = None
        
    @abstractmethod
    def init_element(self):
        pass

    '''
    @abstractmethod
    def calc_stress(self, glob_disps):
        pass

    @abstractmethod
    def update_stiff(self, delta_stress):
        pass
    '''