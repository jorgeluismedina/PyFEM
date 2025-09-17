
import numpy as np

class Node():
    def __init__(self, tag, coord, ndof):
        self.tag = tag
        self.coord = coord
        self.ndof = ndof
        self.set_dof()
        
    def set_dof(self):
        local_dof = np.arange(self.ndof, dtype=int)
        self.dof = self.tag*self.ndof + local_dof
    
