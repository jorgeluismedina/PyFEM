
import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pyfem.femclass import Structure
from pyfem.materials.material import Metal
from pyfem.solvers import tangencial_stiff

#************************************************
#SOLUCION AL PRIMER EJEMPLO DE LA SECCION 3.12.3

#*************************************************

#1. DEFINICION DE MATERIALES, ELEMENTOS, CARGAS Y RESTRICCIONES
etype = 0
idmat = 0
coordenadas = np.array([0, 1, 2, 3, 4, 5])
#elemento_barra = [nodos, area, idmat, etype]
elementos = np.array([[0, 1, 1.0, idmat, etype],
                      [1, 2, 1.0, idmat, etype],
                      [2, 3, 1.0, idmat, etype],
                      [3, 4, 1.0, idmat, etype],
                      [4, 5, 1.0, idmat, etype]])

restricciones = np.array([[0,0,0]])
carga_total = np.array([[0,0,-3],
                        [1,0,-6],
                        [2,0,-6],
                        [3,0,-6],
                        [4,0,-6],
                        [5,0,-3]])

#1. INICIALIZACION DE LA CLASE STRUCTURE
some_metal = Metal(10000, 0.2, 1000, 10, 1.0)#elast, poiss, hards, uniax, dense

structure =  Structure(ndofn=1)
structure.add_materials([some_metal])
structure.add_nodes(coordenadas)
structure.add_elements(elementos)
structure.set_restraints(restricciones)
structure.set_total_loads(carga_total)

#2. RESOLUCION CON EL METODO DE LA RIGIDEZ TANGENCIAL
results = tangencial_stiff(structure, facto=0.04)
displacements, applied_loads = results

#3. PLOTEO DE CURVAS
displacements = np.abs(displacements)
applied_loads = np.abs(applied_loads)
tot_load = np.sum(applied_loads, axis=1)

plt.style.use(['science','notebook','grid'])
fig, ax = plt.subplots(1,1, figsize=(9,6))
plot_style = dict(marker='o', linestyle='-', color='b', markerfacecolor='white', 
                  markersize=5, label='tangencial stiffness method')
ax.plot(displacements[:,-1], tot_load, **plot_style)
ax.set_xlabel("End Displacement (node 6)", fontsize=12)
ax.set_ylabel("Reaction (node 1)", fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=8)
ax.legend(loc='lower right', fancybox=False, edgecolor='black', framealpha=1, fontsize=15)
plt.show()
#plt.savefig("Uniaxial_plasticity.png", dpi=300)
