import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import linalg
from itertools import product
import time
from pyfem.finite_elements import Quad4, Bar2D
from pyfem.fea_system import Structure
from pyfem.gauss_quad import Gauss_Legendre

#***********************************
#EJEMPLO DE ENSAMBLAJE DE MATRICES
#***********************************
materiales = np.array([[2.487e+11, 0.16, 23500],#1
                       [2.5e+11, 0.2, 24000]]) #2

coordenadas = np.array([(-1.0,-0.5),
                        ( 0.0,-0.5),
                        ( 0.5,-0.5),
                        (-1.0, 0.0),
                        ( 0.0, 0.0),
                        ( 0.5, 0.0),
                        (-1.0, 1.0),
                        ( 0.0, 1.0),
                        ( 0.5, 1.0)])
# elemento_membrane = [nodos thick, idmat, etype]
elementos1 = np.array([[0,1,4,3, 0.5, 0, 2],
                       [1,2,5,4, 0.5, 0, 2],
                       [4,5,8,7, 0.5, 0, 2],
                       [3,4,7,6, 0.5, 0, 2]])
# elemento_barra1D = [nodos, area, idmat, etype]
elementos2 = np.array([[0,8, 0.1, 1, 1],
                       [2,6, 0.1, 1, 1]])

restricciones = np.array([[0, 0, 0],
                          [0, 1, 0],
                          [1, 0, 0],
                          [1, 1, 0],
                          [2, 0, 0],
                          [2, 1, 0]])

cargas_puntuales = np.array([[6, 0,  5000],
                             [6, 1, -5000]])

Str1 = Structure([elementos1, elementos2], coordenadas, materiales, 2)
Str1.set_restraints(restricciones)
Str1.set_loads(cargas_puntuales)
Str1.assemble_stiff_mat()

#Prueba de la clase de Gauss_Legendre
scheme = Gauss_Legendre(2,2)
print(scheme.points)

#Prueba de la clase Quad4
elem1 = Str1.elems[0]
print(elem1.shape_funcs(*scheme.points[0])[1])


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

print(check_symmetric(elem1.stiff))
plt.figure()
plt.imshow(elem1.stiff)
#plt.show()

#Prueba de la clase Structure
Str1.solve_system()
print(Str1.gl_disps)

print(check_symmetric(Str1.gl_stiff))
plt.figure()
plt.imshow((Str1.gl_stiff != 0).astype(int), cmap='binary')
plt.show()