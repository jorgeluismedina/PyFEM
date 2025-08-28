
from abc import ABC, abstractmethod
import numpy as np
     
class Node4Shape():
    def funcs(self, r, s):
        # FUNCIONES DE FORMA EN SENTIDO ANTIHORARIO
        # COLOCAR COORDENADAS EN SENTIDO ANTIHORARIO
        N1 = 0.25 * (1 - r) * (1 - s)
        N2 = 0.25 * (1 + r) * (1 - s)
        N3 = 0.25 * (1 + r) * (1 + s)
        N4 = 0.25 * (1 - r) * (1 + s)
        return np.array([N1, N2, N3, N4])
    
    def deriv(self, r, s):
        dN1dr = 0.25 * (-1 + s)
        dN2dr = 0.25 * ( 1 - s)
        dN3dr = 0.25 * ( 1 + s)
        dN4dr = 0.25 * (-1 - s) 

        dN1ds = 0.25 * (-1 + r)
        dN2ds = 0.25 * (-1 - r)
        dN3ds = 0.25 * ( 1 + r)
        dN4ds = 0.25 * ( 1 - r)
        return np.array([[dN1dr, dN2dr, dN3dr, dN4dr],
                         [dN1ds, dN2ds, dN3ds, dN4ds]])


    
class Node8Shape(): #MODOFICAR PARA QUE SEA ANTIHORARIO
    def funcs(self, r, s):
        ss = s*s
        rr = r*r
        N1 = 0.25 * (1 - r) * (1 - s) * (-r - s - 1)
        N2 = 0.50 * (1 - r) * (1 - ss)
        N3 = 0.25 * (1 - r) * (1 + s) * (-r + s - 1)
        N4 = 0.50 * (1 - rr) * (1 + s)
        N5 = 0.25 * (1 + r) * (1 + s) * (r + s - 1)
        N6 = 0.50 * (1 + r) * (1 - ss)
        N7 = 0.25 * (1 + r) * (1 - s) * (r - s - 1)
        N8 = 0.50 * (1 - rr) * (1 - s)
        return np.array([N1, N2, N3, N4, N5, N6, N7, N8])
    
    def deriv(self, r, s):
        ss = s*s
        rr = r*r
        s2 = s*2
        r2 = r*2

        dN1dr = 0.25 * ( 1 - s) * (r2 + s)
        dN2dr = 0.50 * (-1 + ss)
        dN3dr = 0.25 * ( 1 + s) * (r2 - s)
        dN4dr = r * (-1 - s)
        dN5dr = 0.25 * ( 1 + s) * (r2 + s)
        dN6dr = 0.50 * ( 1 - ss)
        dN7dr = 0.25 * ( 1 - s) * (r2 - s)
        dN8dr = r * (-1 + s)

        dN1ds = 0.25 * ( 1 - r) * (r + s2)
        dN2ds = s * (-1 + r)
        dN3ds = 0.25 * (-1 + r) * (r - s2)
        dN4ds = 0.50 * ( 1 - rr)
        dN5ds = 0.25 * ( 1 + r) * (r + s2)
        dN6ds = s * (-1 - r)
        dN7ds = 0.25 * (-1 - r) * (r - s2)
        dN8ds = 0.50 * (-1 + rr)
        return np.array([[dN1dr, dN2dr, dN3dr, dN4dr, dN5dr, dN6dr, dN7dr, dN8dr],
                         [dN1ds, dN2ds, dN3ds, dN4ds, dN5ds, dN6ds, dN7ds, dN8ds]])