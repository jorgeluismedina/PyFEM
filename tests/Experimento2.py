import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from pyfem.femclass import Structure
from pyfem.gauss_quad import Gauss_Legendre
from pyfem.materials.material import Metal
from pyfem.solvers import check_symmetric

#***********************************
#EJEMPLO DE ENSAMBLAJE DE MATRICES
#***********************************


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

carga_total = np.array([[6, 0,  5000],
                        [6, 1, -5000]])

metal1 = Metal(2.487e+11, 0.16, 0.0, 16000, 1.0)#elast, poiss, hards, uniax, dense
metal2 = Metal(2.500e+11, 0.20, 0.0, 16000, 1.0)#elast, poiss, hards, uniax, dense
metal1.add_constitutive_model('PlaneStress')
metal1.add_yield_criterion('Tresca')
materiales = [metal1, metal2]

structure = Structure(ndofn=2)
structure.add_materials(materiales)
structure.add_nodes(coordenadas)
structure.add_elements(elementos1)
structure.add_elements(elementos2)
structure.set_restraints(restricciones)
structure.set_total_loads(carga_total)


#Prueba de la clase de Gauss_Legendre
scheme = Gauss_Legendre(3,2)
print(scheme.points)
print(scheme.weights)

glob_stiff = structure.assemb_global_stiff()

print('Elements stiffness')
for i, e in enumerate(structure.elems):
    print(e.stiff)
    print('symmetric: ', check_symmetric(e.stiff))

print('\nGlobal Stiffnes\n')
print(glob_stiff)
print('symmetric: ', check_symmetric(glob_stiff))

plt.figure()
plt.imshow((abs(glob_stiff) > 1e-6), cmap='binary')
plt.show()