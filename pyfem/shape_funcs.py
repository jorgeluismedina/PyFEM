
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
        shape = self.funcs(r,s)
        N = np.zeros((self.ndofn, self.ndofn*shape.shape[0]))
        N[0, 0::2] = shape
        N[1, 1::2] = shape
        return N
    
    
class Node4Shape(ShapeFuncs):
    def funcs(self, r, s):
        N1 = (1 - r) * (1 - s) * 0.25
        N2 = (1 - r) * (1 + s) * 0.25
        N3 = (1 + r) * (1 + s) * 0.25
        N4 = (1 + r) * (1 - s) * 0.25
        return np.array([N1, N2, N3, N4])
    
    def deriv(self, r, s):
        dN1dr = (-1 + s) * 0.25
        dN2dr = (-1 - s) * 0.25
        dN3dr = ( 1 + s) * 0.25
        dN4dr = ( 1 - s) * 0.25
        dN1ds = (-1 + r) * 0.25
        dN2ds = ( 1 - r) * 0.25
        dN3ds = ( 1 + r) * 0.25
        dN4ds = (-1 - r) * 0.25
        return np.array([[dN1dr, dN2dr, dN3dr, dN4dr],
                         [dN1ds, dN2ds, dN3ds, dN4ds]])


    
class Node8Shape(ShapeFuncs):
    def funcs(self, r, s):
        ss = s*s
        rr = r*r
        N1 = (1 - r) * (1 - s) * (-r - s - 1) * 0.25
        N2 = (1 - r) * (1 - ss) * 0.5
        N3 = (1 - r) * (1 + s) * (-r + s - 1) * 0.25
        N4 = (1 - rr) * (1 + s) * 0.5
        N5 = (1 + r) * (1 + s) * (r + s - 1) * 0.25
        N6 = (1 + r) * (1 - ss) * 0.5
        N7 = (1 + r) * (1 - s) * (r - s - 1) * 0.25
        N8 = (1 + rr) * (1 - s) * 0.5
        return np.array([N1, N2, N3, N4, N5, N6, N7, N8])
    
    def deriv(self, r, s):
        ss = s*s
        rr = r*r
        s2 = s*2
        r2 = r*2
        dN1dr = ( 1 - s) * (r2 + s) * 0.25
        dN2dr = (-1 + ss) * 0.5
        dN3dr = ( 1 + s) * (r2 - s) * 0.25
        dN4dr = r * (-1 - s)
        dN5dr = ( 1 + s) * (r2 + s) * 0.25
        dN6dr = ( 1 - ss) * 0.5
        dN7dr = ( 1 - s) * (r2 - s) * 0.25
        dN8dr = r * ( 1 - s)
        dN1ds = ( 1 - r) * (r + s2) * 0.25
        dN2ds = s * (-1 + r)
        dN3ds = (-1 + r) * (r - s2) * 0.25
        dN4ds = ( 1 - rr) * 0.5
        dN5ds = ( 1 + r) * (r + s2) * 0.25
        dN6ds = s * (-1 - r)
        dN7ds = ( -1 - r) * (r - s2) * 0.25
        dN8ds = (-1 - rr) * 0.5
        return np.array([[dN1dr, dN2dr, dN3dr, dN4dr, dN5dr, dN6dr, dN7dr, dN8dr],
                         [dN1ds, dN2ds, dN3ds, dN4ds, dN5ds, dN6ds, dN7ds, dN8ds]])