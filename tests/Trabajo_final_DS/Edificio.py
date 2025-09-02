
import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import scipy as sp
import pickle
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt

from pyfem.femclass import Model
from pyfem.materials.material import Material
from pyfem.sections import FrameSection
from pyfem.plotting import print_matrix, plot_2dmodel, plot_matrix
from pyfem.solvers import solve_linear_static, vibration_modes, finite_differences

# Materiales
# Para problemas en dinamica el modulo de elasticidad tiene que estar en [N/m2]
steel = Material(elast=2.57e10, poiss=0.2, dense=2500) #[N/m2] [] [Kg/m3]
materials = [steel]

# Secciones
a = 0.2
b = 0.8
sect1 = FrameSection(xarea=a*b, inrt3=a*b**3/12) #[m2]
sections = [sect1]

mod = Model(ndofn=3)
mod.add_materials(materials)
mod.add_sections(sections)

nodos = []
# LECTURA DE DATOS
with open(r'D:\ProyectosPy\PyFEM\tests\Trabajo_final_DS\Edificio3.msh', 'r') as file:
    lines = file.readlines()

part_coordinates = False
part_elements = False

for line in lines:
    line = line.strip()

    if line.startswith('Coordinates'):
        part_coordinates = True
        continue
    elif line.startswith('End Coordinates'):
        part_coordinates = False
        mod.add_nodes(np.array(nodos))
        continue
    elif line.startswith('Elements'):
        part_elements = True
        continue
    elif line.startswith('End Elements'):
        part_elements = False
        continue 

    if part_coordinates:
        data = line.split()
        x = float(data[1])
        y = float(data[2])
        nodos.append([x,y])

    elif part_elements:
        data = line.split()
        tag = int(data[0])-1
        ni = int(data[1])-1
        nj = int(data[2])-1
        mod.add_element(tag, [ni, nj], sect1, steel, 'Frame22D')
        #elements.append([tag, ni, nj])
        

#mod.add_nodes(np.array(nodos))
#for element in elements:
#    tag, ni, nj = element
#    mod.add_element(tag, [ni, nj], sect1, steel, 'Truss2D')

#mod.add_node_restraint(20, [1, 1, 1])
#mod.add_node_restraint(21, [1, 1, 1])
mod.add_node_restraint(134, [1, 1, 1])
mod.add_node_restraint(132, [1, 1, 1])
mod.add_node_restraint(130, [1, 1, 1])
mod.add_node_restraint(133, [1, 1, 1])
mod.add_node_restraint(135, [1, 1, 1])
mod.add_node_restraint(137, [1, 1, 1])
mod.add_node_restraint(139, [1, 1, 1])
mod.add_node_restraint(141, [1, 1, 1])
mod.add_node_restraint(142, [1, 1, 1])



mod.set_restraints()
# Matrices reducidas
K = mod.assemb_global_stiff()[np.ix_(mod.free_dof, mod.free_dof)] #[Nm]
M = mod.assemb_global_mass()[np.ix_(mod.free_dof, mod.free_dof)] #[Kg]

#SOLUCION
fk, wk, Phi = vibration_modes(K, M)
print('Frecuencia Modo 1: {0:.4f} Hz'.format(fk[0]))
print('Frecuencia Modo 2: {0:.4f} Hz'.format(fk[1]))
print('Frecuencia Modo 3: {0:.4f} Hz'.format(fk[2]))

#-------------------------------------------------------------------
# Espectros
def generar_espectro(wg, xig):
    w = np.linspace(0.1, 10*2*np.pi, 200)
    S0 = 0.03*xig / (np.pi*wg * (4 * xig**2 + 1))
    aux = 4 * wg**2 * xig**2 * w**2
    S = S0 * (wg**4 + aux) / ((w**2-wg**2)**2 + aux)
    return S, w

#S_soft, w = generar_espectro(2.4*np.pi, 0.85) #Azul
#S_stif, w = generar_espectro(5.0*np.pi, 0.60) #Rojo

#------------------------------------------------------------------------
# Generar aceleraciones
def generar_aceleracion(wg, xig, tspan, dt):
    # espectro
    w = np.linspace(0.1, 10*2*np.pi, 200)
    S0 = 0.03*xig / (np.pi*wg * (4 * xig**2 + 1))
    aux = 4 * wg**2 * xig**2 * w**2
    S = S0 * (wg**4 + aux) / ((w**2-wg**2)**2 + aux)

    t = np.arange(0, tspan, dt)
    Nw = len(w) #intervalos de frecuencia
    dw = w[1]-w[0]
    phi = np.random.uniform(0, 2*np.pi, Nw) 
    A = np.sqrt(2 * S * dw)
    acc = np.zeros_like(t)
    for j in range(Nw):
        acc += A[j] * np.cos(w[j] * t + phi[j])
    return acc

acc1 = 9.81*generar_aceleracion(5.0*np.pi, 0.60, tspan=20, dt=0.01) # Stiff rojo
acc2 = 9.81*generar_aceleracion(2.4*np.pi, 0.85, tspan=25, dt=0.01) # Soft azul

print(acc1.shape)
print(acc2.shape)
print((np.arange(0, 50, 0.01)).shape)


def generate_signal(acc, tspan, dt, t_start): 
    t = np.arange(0, tspan, dt)
    ag = np.zeros_like(t)
    idx_start = int(t_start / dt)
    idx_end = idx_start + acc.shape[0]
    ag[idx_start:idx_end] = acc
    return ag, t, idx_start, idx_end

ag1, t, i_start1, i_end1 = generate_signal(acc1, tspan=50, dt=0.01, t_start=5) # Stiff rojo
ag2, t, i_start2, i_end2 = generate_signal(acc2, tspan=50, dt=0.01, t_start=5) # Soft azul
print(i_start1, i_end1)
print(i_start2, i_end2)
print(t[i_start1], t[i_end1])
print(t[i_start2], t[i_end2])


fig2 = plt.figure()
plt.plot(t, ag1, 'r', label='stiff soil')
plt.plot(t, ag2, 'b', label='soft soil')
plt.legend()
plt.show()

#------------------------------------------------------------------------------
# Creacion del vector de fuerzas

def seismic_force(model, ag):
    Mglob = model.assemb_global_mass()
    xdofs = model.all_dof[0::3]
    A = np.zeros(model.ndofs)
    nsteps = ag.shape[0]
    F = np.zeros((nsteps, len(model.free_dof)))

    for step in range(nsteps):
        A[xdofs] = ag[step] 
        Fglob = Mglob @ A
        f = Fglob[model.free_dof]
        f[np.abs(f)<1e-6] = 0.0
        F[step] = f
    return F

F1 = seismic_force(mod, ag1)
F2 = seismic_force(mod, ag2)


#--------------------------------------------------------------------------------
#Creacion de la matriz C
wki = wk[0];  zti = 0.01
wkj = wk[1];  ztj = 0.01

alpha = np.linalg.solve([[1/(2*wki), wki/2], 
                         [1/(2*wkj), wkj/2]], [zti, ztj])

C = alpha[0]*M + alpha[1]*K #Ya esta reducida
print('alpha0: {0:.4f} s'.format(alpha[0]))
print('alpha1: {0:.4f} s'.format(alpha[1]))


#------------------------------------------------------------------------------------
# Newmark

def Newmark(F, K, M, C, free_dof, ndofs, dt=0.01):

    gamma = 0.5
    beta = 0.25

    nsteps = F.shape[0]
    nfreedof = len(free_dof)
    disp = np.zeros((nsteps, nfreedof))
    velo = np.zeros((nsteps, nfreedof))
    acce = np.zeros((nsteps, nfreedof))

    glob_disp = np.zeros((nsteps, ndofs))
    glob_velo = np.zeros((nsteps, ndofs))
    glob_acce = np.zeros((nsteps, ndofs))

    acce[0] = sp.linalg.solve(M, F[0], assume_a='sym')

    # Constantes
    a0 = 1 / (beta * dt ** 2)
    a1 = gamma / (beta * dt)
    a2 = 1 / (beta * dt)
    a3 = 1 / (2 * beta) - 1
    a4 = gamma / beta - 1
    a5 = (dt / 2) * (gamma / beta - 2)

    K_eff = K + a0 * M + a1 * C
    chol = cho_factor(K_eff)

    
    for step in range(nsteps-1):
        F_eff = (F[step+1]
                + M @ (a0 * disp[step] + a2 * velo[step] + a3 * acce[step])
                + C @ (a1 * disp[step] + a4 * velo[step] + a5 * acce[step]))
        
        disp[step+1] = cho_solve(chol, F_eff)
        acce[step+1] = (a0 * (disp[step+1] - disp[step])
                                - a2 * velo[step] - a3 * acce[step])
        velo[step+1] = (velo[step]
                                + (1 - gamma) * dt * acce[step]
                                + gamma * dt * acce[step+1])
        

    glob_disp[:, free_dof] = disp
    glob_velo[:, free_dof] = velo
    glob_acce[:, free_dof] = acce

    return glob_disp, glob_velo, glob_acce

glob_disp1, glob_velo1, glob_acce1 = Newmark(F1, K, M, C, mod.free_dof, mod.ndofs) # Stiff rojo
glob_disp2, glob_velo2, glob_acce2 = Newmark(F2, K, M, C, mod.free_dof, mod.ndofs) # Soft azul

print('numero de elementos: {0}'.format(len(mod.elems)))
print('numero de nodos: {0}'.format(len(mod.coord)))

#'''
root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
ruta_pickle1 = os.path.join(root, "Trabajo_final_DS", "sismos.pkl")
ruta_pickle2 = os.path.join(root, "Trabajo_final_DS", "matrices.pkl")
ruta_pickle3 = os.path.join(root, "Trabajo_final_DS", "resultados1.pkl")
ruta_pickle4 = os.path.join(root, "Trabajo_final_DS", "resultados2.pkl")

sismos = (ag1, ag2)
with open(ruta_pickle1, "wb") as f:
    pickle.dump(sismos, f)

matrices = (wk, Phi, M, K, C)
with open(ruta_pickle2, "wb") as f:
    pickle.dump(matrices, f)

resultados1 = (t, glob_disp1, glob_velo1, glob_acce1)
with open(ruta_pickle3, "wb") as f:
    pickle.dump(resultados1, f)

resultados2 = (t, glob_disp2, glob_velo2, glob_acce2)
with open(ruta_pickle4, "wb") as f:
    pickle.dump(resultados2, f)

#'''