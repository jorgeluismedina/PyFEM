
import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pyfem.fea_system import Structure
from pyfem.materials.material import Metal
from pyfem.solvers import tangencial_stiff

#************************************************
#SOLUCION AL

#*************************************************

#1. DEFINICION DE MATERIALES, ELEMENTOS, CARGAS Y RESTRICCIONES
etype = 2
idmat = 0
width = 0.5

coordenadas = np.array([[0.0, 0.0],
                        [2.5, 0.0],
                        [5.0, 0.0],
                        [7.5, 0.0],
                        [10.0, 0.0],
                        [0.0, 1.0],
                        [2.5, 1.0],
                        [5.0, 1.0],
                        [7.5, 1.0],
                        [10.0, 1.0]])

#elemento_quad4 = [nodos, thick, idmat, idtype]
elementos = np.array([[0, 1, 6, 5, width, idmat, etype],
                      [1, 2, 7, 6, width, idmat, etype],
                      [2, 3, 8, 7, width, idmat, etype],
                      [3, 4, 9, 8, width, idmat, etype]])

restricciones = np.array([[0,0,0],
                          [0,1,0],
                          [4,0,0],
                          [4,1,0]])

q = -800
carga_total = np.array([[5,1,q*1.25],
                        [6,1,q*2.50],
                        [7,1,q*2.50],
                        [8,1,q*2.50],
                        [9,1,q*1.25]])

#1. INICIALIZACION DE LA CLASE STRUCTURE
some_metal = Metal(10e6, 0.24, 0.0, 16000, 1.0)#elast, poiss, hards, uniax, dense
some_metal.add_constitutive_model('PlaneStress')
some_metal.add_yield_criterion('VonMises')

structure =  Structure(ndofn=2)
structure.add_materials([some_metal])
structure.add_nodes(coordenadas)
structure.add_elements(elementos)
structure.set_restraints(restricciones)
structure.set_total_loads(carga_total)



#2. RESOLUCION CON EL METODO DE LA RIGIDEZ TANGENCIAL
results = tangencial_stiff(structure, facto=0.06)
displacements, applied_loads = results



#3. PLOTEO DE CURVAS
# Se cambia el signo de los desplazamientos y cargas a positivo
# ya que como tienen la misma direccion la grafica se ve mejor en positivo
displacements = np.abs(displacements)
applied_loads = np.abs(applied_loads)

dof5_disp = displacements[:,5]
dof15load = applied_loads[:,15]

plt.style.use(['science','notebook','grid'])
fig, ax = plt.subplots(1,1, figsize=(9,6))
plot_style = dict(marker='o', linestyle='-', color='b', markerfacecolor='white', 
                  markersize=5, label='tangencial stiffness method')

ax.plot(dof5_disp, dof15load, **plot_style)
ax.set_xlabel("uy-displacement (node 2)", fontsize=12)
ax.set_ylabel("Load (node 7)", fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=8)
ax.legend(loc='lower right', fancybox=False, edgecolor='black', framealpha=1, fontsize=15)
plt.show()
#'''