
import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pyfem.fea_system import Structure
from pyfem.materials import Metal

#SOLUCION AL PROBLEMA ELASTO-PLASTICO
#************************************

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
cargas_fijas = np.array([[0,0,0]])
cargas_incre = np.array([[0,0,3],
                         [1,0,6],
                         [2,0,6],
                         [3,0,6],
                         [4,0,6],
                         [5,0,3]])


some_metal = Metal(10000, 0.2, 1000, 10, 1.0)#elast, poiss, hards, uniax, dense
some_metal.add_constitutive_model('PlaneStress')
some_metal.add_yield_criterion('Tresca')
Sys = Structure([elementos], coordenadas, [some_metal], 1)
Sys.set_restraints(restricciones)
Sys.set_loads(cargas_fijas, cargas_incre, nincs=20)

#print(Sys.gl_loads)



# RESOLUCION CON EL METODO DE RESOLUCION TANGENCIAL

displacements = []
elem_stress = []
phi = np.zeros(Sys.fr_dof.shape[0])
H = Sys.retan_stiff(phi)
max_iter = 10
tol = 1e-5
kount = 1

for f in -Sys.re_loads:
    unbalance = False
    #elem_stress.append([elem.stress for elem in Sys.elems])
    for i in range(max_iter):
        psi = Sys.residual_forces(H, phi, f)
        if sp.linalg.norm(psi) < tol:
            #print("convergence reached")
            break
        unbalance = True
        #print("unbalanced forces")
        delta_phi = -sp.linalg.solve(H, psi)
        phi = phi + delta_phi
        H = Sys.retan_stiff(phi)
        
    print(f"convergence reached in iterations: {i}")       
    if not unbalance:
        H = Sys.retan_stiff(phi)

    displacements.append(phi)
    kount += 1

#print(elem_stress)
# GRAFICO
displacements = np.array(displacements)
cum_disp = np.cumsum(displacements, axis=0)
cum_load = np.cumsum(Sys.gl_loads, axis=0)
tot_load = np.sum(np.cumsum(Sys.gl_loads, axis=0), axis=1)

plt.style.use(['science','notebook','grid'])
fig, ax = plt.subplots(1,1, figsize=(9,6))
plot_style = dict(marker='o', linestyle='-', color='b', markerfacecolor='white', 
                  markersize=5, label='tangencial stiffness method')
ax.plot(cum_disp[:,-1], tot_load, **plot_style)
ax.set_xlabel("End Displacement (node 6)", fontsize=12)
ax.set_ylabel("Reaction (node 1)", fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=8)
ax.legend(loc='lower right', fancybox=False, edgecolor='black', framealpha=1, fontsize=15)
plt.show()
#plt.savefig("Uniaxial_plasticity.png", dpi=300)
