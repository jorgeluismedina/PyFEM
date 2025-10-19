
import numpy as np
from abc import ABC, abstractmethod

class Element():
    # def __init__(self, conec, dof, coord, mater):
    def __init__(self, mater, coord, conec, dof): # el atributo conec no lo uso para nada
        self.mater = mater
        self.coord = coord
        self.conec = conec
        self.dof = dof
        #self.conec = conec
        #self.section = section
        #self.nnods = conec.shape[0]
        self.loads = None
        
    #@abstractmethod
    #def init_element(self):
    #    pass

        '''
    @abstractmethod
    def calc_stress(self, glob_disps):
        pass

    @abstractmethod
    def update_stiff(self, delta_stress):
        pass
    '''


class FrameElement(Element):
    def __init__(self, mater, section, coord, conec, dof):
        super().__init__(mater, coord, conec, dof)
        self.section = section


class AreaElement(Element):
    def __init__(self, mater, section, coord, conec, dof):
        super().__init__(mater, coord, conec, dof)
        self.section = section

