
import numpy as np
from abc import ABC, abstractmethod

class Element(ABC):
    def __init__(self, nodes, coord, mater): 
        self.coord = coord
        self.nodes = nodes
        self.mater = mater
        self.nnods = nodes.shape[0]
        self.eload = None
        self.dof = None
        
    def set_dof(self, ndofn: int) -> None:
        self.ndofn = ndofn
        node_dof = np.arange(ndofn, dtype=int)
        self.dof = np.repeat(ndofn*self.nodes, ndofn) + np.tile(node_dof, self.nnods)
    
    @abstractmethod
    def set_stiff_mat(self):
        pass

    @abstractmethod
    def calc_stress(self, u):
        pass

    @abstractmethod
    def update_elem(self, stress):
        pass