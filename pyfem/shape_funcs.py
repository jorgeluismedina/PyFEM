
from abc import ABC, abstractmethod
import numpy as np

class ShapeFuncs(ABC):
    def __init__(self, ndofn):
        self.ndofn = ndofn

    @abstractmethod
    def funcs(self, r, s):
        pass

    @abstractmethod
    def deriv(self, r, s):
        pass

    def matrix(self, r, s):
        N = self.funcs(r,s)
        H = np.zeros((self.ndofn, self.ndofn*N.shape[0]))
        H[0, 0::2] = N
        H[1, 1::2] = N
        return H
    
    
class Node4Shape(ShapeFuncs):
    def funcs(self, r, s):
        N1 = (1 - r)*(1 - s) * 0.25
        N2 = (1 - r)*(1 + s) * 0.25
        N3 = (1 + r)*(1 + s) * 0.25
        N4 = (1 + r)*(1 - s) * 0.25
        return np.array([N1, N2, N3, N4])
    
    def deriv(self, r, s):
        dN1dr = ( s - 1) * 0.25
        dN2dr = (-s - 1) * 0.25
        dN3dr = ( s + 1) * 0.25
        dN4dr = (-s + 1) * 0.25
        dN1ds = ( r - 1) * 0.25
        dN2ds = (-r + 1) * 0.25
        dN3ds = ( r + 1) * 0.25
        dN4ds = (-r - 1) * 0.25
        return np.array([[dN1dr, dN2dr, dN3dr, dN4dr],
                         [dN1ds, dN2ds, dN3ds, dN4ds]])