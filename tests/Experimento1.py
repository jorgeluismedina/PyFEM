
import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pyfem.fea_system import Structure

#SOLUCION AL PROBLEMA ELASTO-PLASTICO
#************************************

#1. DEFINICION DE MATERIALES, ELEMENTOS, CARGAS Y RESTRICCIONES
etype = 0
idmat = 0
# material = [E, nu, sy, H', dens]
material = np.array([[1000, 0.2, 40, 100, 1.0]])
coordenadas = np.array([0, 1, 2])
#elemento_barra = [nodos, area, idmat, etype]
elementos = np.array([[0,1, 2.0, idmat, etype],
                      [1,2, 1.0, idmat, etype]])
restricciones = np.array([[0,0,0]])
cargas_fijas = np.array([[0,0,0]])
cargas_incre = np.array([[2,0,150]])

#Creacion de clases
Sys1 = Structure([elementos], coordenadas, material, 1)
Sys1.set_restraints(restricciones)
Sys1.set_loads(cargas_fijas, cargas_incre, nincs=20)
print(Sys1.gl_loads)

#2. RESOLUCION Y ACTUALIZACION DIRECTA
displacements = []
elem_stress = []
kount = 1
for f in Sys1.re_loads:
    print (f"load step: {kount}")
    Sys1.assemble_stiff_mat()
    print(f"Matriz de Rigidez:\n {Sys1.gl_stiff}")
    du = Sys1.direct_solve(Sys1.re_stiff, f)
    displacements.append(du)
    print(f"desplazamiento: {du}")
    Sys1.get_element_stresses(du)
    elemstrs = [elem.stress for elem in Sys1.elems]
    print(f"Esfuerzos: {elemstrs}")
    elem_stress.append(elemstrs)
    kount += 1

displacements = np.array(displacements)
cum_disp = np.cumsum(displacements, axis=0)
cum_load = np.cumsum(Sys1.gl_loads, axis=0)
print(np.hstack([displacements, cum_disp]))

plt.figure()
plt.plot(cum_disp[:,1], cum_load[:,-1],'o-', label="node 2")
plt.plot(cum_disp[:,2], cum_load[:,-1],'o-', label="node 3")
plt.xlabel("displacements")
plt.ylabel("load")
plt.legend()


#3. RESOLUCION CON EL METODO DE RESOLUCION TANGENCIAL
Sys2 = Structure([elementos], coordenadas, material, 1)
Sys2.set_restraints(restricciones)
Sys2.set_loads(cargas_fijas, cargas_incre, nincs=20)

def residual(H, phi, f):
    return H @ phi + f

def tangencial_stiffness(H_func, f, phi0, tol=1e-5, max_iter=100):
    phi = phi0
    for i in range(max_iter):
        H = H_func(phi)
        psi = residual(H, phi, f)
        if sp.linalg.norm(psi) < tol:
            break
        delta_phi = -sp.linalg.solve(H, psi)
        phi = phi + delta_phi
    else:
        print('Convergence not reached')
    return phi, i

displacements2 = []
elem_stress2 = []
guess = np.zeros(Sys2.fr_dof.shape[0])
kount = 1
#stress_e2 = []
for f in Sys2.re_loads:
    print(f"Load Step:{kount}")
    du, ite = tangencial_stiffness(Sys2.retan_stiff, -f, guess, max_iter=10)
    displacements2.append(du)
    print(f"desplazamiento: {du}, iteraciones: {ite}")
    elemstrs2 = [elem.stress for elem in Sys2.elems]
    print(f"Esfuerzos: {elemstrs2}")
    elem_stress2.append(elemstrs2)
    guess = du
    kount += 1

displacements2 = np.array(displacements2)
cum_disp2 = np.cumsum(displacements2, axis=0)
cum_load2 = np.cumsum(Sys2.gl_loads, axis=0)
print(np.hstack([displacements2, cum_disp2]))

plt.figure()
plt.plot(cum_disp2[:,1], cum_load2[:,-1],'o-', label="tangencial stiffness node 3")
plt.plot(cum_disp2[:,0], cum_load2[:,-1],'o-', label="tangencial stiffness node 2")
plt.plot(cum_disp[:,2], cum_load[:,-1],'o-', label="direct solve node 3")
plt.plot(cum_disp[:,1], cum_load[:,-1],'o-', label="direct solve node 2")
plt.xlabel("displacements")
plt.ylabel("load")
plt.legend()
plt.title('Fig 2')
plt.show()