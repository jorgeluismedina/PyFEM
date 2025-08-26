
import numpy as np
from abc import ABC, abstractmethod

class Element(ABC):
    def __init__(self, nodes, coord, section, mater): # Que no tenga el atributo section
        self.coord = coord
        self.nodes = nodes
        self.mater = mater
        self.section = section
        self.nnods = nodes.shape[0]
        self.loads = None
        self.dof = None
        
    def set_dof(self, ndofn: int):
        self.ndofn = ndofn
        node_dof = np.arange(ndofn, dtype=int)
        self.dof = np.repeat(ndofn*self.nodes, ndofn) + np.tile(node_dof, self.nnods)
    
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