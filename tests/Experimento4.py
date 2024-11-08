
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
etype = 3
idmat = 0

nelem = 4
beam_length = 4
beam_height = 0.4
beam_width = 0.2

lim1 = np.array([[0.0, 0.0],[0.0, beam_height/2],[0.0, beam_height]])
lim2 = lim1+[[beam_length, 0.0]]
infrow = np.linspace(lim1[0], lim2[0], 2*nelem+1, axis=0)
midrow = np.linspace(lim1[1], lim2[1], nelem+1, axis=0)
suprow = np.linspace(lim1[2], lim2[2], 2*nelem+1, axis=0)
coordenadas = np.vstack([infrow, midrow, suprow])

inftag = np.arange(infrow.shape[0])
midtag = np.arange(inftag[-1]+1, infrow.shape[0]+midrow.shape[0])
suptag = np.arange(midtag[-1]+1, coordenadas.shape[0])
print(inftag)
print(midtag)
print(suptag)
print(coordenadas.shape)

#elemento_quad4 = [nodos, thick, idmat, idtype]
elementos = np.array([[0, 1, 2, 10, 16, 15, 14, 9, beam_width, idmat, etype],
                      [2, 3, 4, 11, 18, 17, 16, 10, beam_width, idmat, etype],
                      [4, 5, 6, 12, 20, 19, 18, 11, beam_width, idmat, etype],
                      [6, 7, 8, 13, 22, 21, 20, 12, beam_width, idmat, etype]])
'''
elementos = np.array([[0, 1, 2, 6, 10, 9, 8, 5, beam_width, idmat, etype],
                      [2, 3, 4, 7, 12, 11, 10, 6, beam_width, idmat, etype]])


restricciones = np.array([[0,0,0],
                          [0,1,0],
                          [4,1,0]])

'''

restricciones = np.array([[0,0,0],
                          [0,1,0],
                          [8,1,0]])

distr_load = -250
nodal_load = np.full_like(suptag, distr_load*beam_length/nelem/2)
nodal_load[0] = nodal_load[0]/2
nodal_load[-1] = nodal_load[-1]/2
carga_total = np.vstack([suptag, np.ones_like(suptag), nodal_load]).T
'''
carga_total = np.array([[14,1,q*beam_length/nelem/2/2],
                        [15,1,q*beam_length/nelem/2],
                        [16,1,q*beam_length/nelem/2],
                        [17,1,q*beam_length/nelem/2],
                        [18,1,q*beam_length/nelem/2],
                        [19,1,q*beam_length/nelem/2],
                        [20,1,q*beam_length/nelem/2],
                        [21,1,q*beam_length/nelem/2],
                        [22,1,q*beam_length/nelem/2/2]])
'''



#1. INICIALIZACION DE LA CLASE STRUCTURE
some_metal = Metal(1.755e7, 0.2, 0.0, 16000, 1.0)#elast, poiss, hards, uniax, dense
some_metal.add_constitutive_model('PlaneStress')
some_metal.add_yield_criterion('VonMises')

structure = Structure(ndofn=2)
structure.add_materials([some_metal])
structure.add_nodes(coordenadas)
structure.add_elements(elementos)
structure.set_restraints(restricciones)
structure.set_total_loads(carga_total)


#'''
#2. RESOLUCION CON EL METODO DE LA RIGIDEZ TANGENCIAL
results = tangencial_stiff(structure, facto=0.05)
displacements, applied_loads = results


#3. PLOTEO DE CURVAS
# Se cambia el signo de los desplazamientos y cargas a positivo
# ya que como tienen la misma direccion la grafica se ve mejor en positivo
displacements = np.abs(displacements)
distribu_load = np.sum(np.abs(applied_loads), axis=1)/beam_length

dof9_disp = displacements[:,9]

plt.style.use(['science','notebook','grid'])
fig, ax = plt.subplots(1,1, figsize=(9,6))
plot_style = dict(marker='o', linestyle='-', color='b', markerfacecolor='white', 
                  markersize=5, label='tangencial stiffness method')

ax.plot(dof9_disp, distribu_load, **plot_style)
ax.set_xlabel("Uy-displacement (node 4)", fontsize=12)
ax.set_ylabel("Distributed Load", fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=8)
ax.legend(loc='lower right', fancybox=False, edgecolor='black', framealpha=1, fontsize=15)
plt.show()
#'''