
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
steel = Material(elast=2.57e10, poiss=0.2, dense=2500) #[KN/m2] [] [Kg/m3]
materials = [steel]

# Secciones
a = 0.2
b = 0.7
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
print('Frecuencia fundamental: {0:.4f} Hz'.format(fk[0]))

#-------------------------------------------------------------------
# Espectros
def generar_espectro(w, wg, xig):
    S0 = 0.03*xig / (np.pi*wg * (4 * xig**2 + 1))
    aux = 4 * wg**2 * xig**2 * w**2
    return S0 * (wg**4 + aux) / ((w**2-wg**2)**2 + aux)

w = np.linspace(0.1, 10*2*np.pi, 200)
S_soft = generar_espectro(w, 2.4*np.pi, 0.85)
S_stif = generar_espectro(w, 5.0*np.pi, 0.60)

#fig1 = plt.figure()
#plt.plot(w/(2*np.pi), S_stif/(4*np.pi**2), 'b', label='stiff soil')
#plt.plot(w/(2*np.pi), S_soft/(4*np.pi**2), 'g', label='soft soil')
#plt.xlim(0,10)
#plt.legend()

#------------------------------------------------------------------------
# Generar aceleraciones
def generar_aceleracion(S, w, t):
    Nw = len(w) #intervalos de frecuencia
    dw = w[1]-w[0]
    phi = np.random.uniform(0, 2*np.pi, Nw) 
    A = np.sqrt(2 * S * dw)
    acc = np.zeros_like(t)
    for j in range(Nw):
        acc += A[j] * np.cos(w[j] * t + phi[j])

    return acc


tspan_before = 5
tspan_sismo = 25 #[s]
tspan_after = 25
t_total = tspan_before + tspan_sismo + tspan_after
dt = 0.01
N_antes = int(tspan_before / dt)
N_sismo = int(tspan_sismo / dt)
N_after = int(tspan_after / dt)
#N_total = int(len(t_total))
tb = np.linspace(0, tspan_before, N_antes, endpoint=False)
ts = np.linspace(tspan_before, tspan_before+tspan_sismo, N_sismo, endpoint=False)
ta = np.linspace(tspan_before+tspan_sismo, t_total, N_after, endpoint=False)
ttot = np.hstack([tb, ts, ta])
start_sism_idx = tb.shape[0]-1
end_sism_idx = start_sism_idx + ts.shape[0]-1
print('sism start idx: {0}'.format(start_sism_idx))
print('sism end idx: {0}'.format(end_sism_idx))

ag1 = 9.81 * generar_aceleracion(S_stif, w, ts)
ag2 = 9.81 * generar_aceleracion(S_soft, w, ts)

agtot = np.hstack([np.zeros(N_antes), 
                   ag1, 
                   np.zeros(int(tspan_after/(dt)))])

#fig2 = plt.figure()
#plt.plot(t, ag1, 'b', label='stiff soil')
#plt.plot(ttot, agtot, 'r', label='soft soil')
#plt.legend()

#------------------------------------------------------------------------------
# Creacion de la matriz de fuerzas
nsteps = agtot.shape[0]
A = np.zeros(mod.ndofs)
F = np.zeros((nsteps, len(mod.free_dof)))
Mglob = mod.assemb_global_mass()
xdofs = mod.all_dof[0::3]
for step in range(nsteps):
    A[xdofs] = agtot[step] 
    Fglob = Mglob @ A
    f = Fglob[mod.free_dof]
    f[np.abs(f)<1e-6] = 0.0
    F[step] = f


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
gamma = 0.5
beta = 0.25

nfreedof = len(mod.free_dof)
disp = np.zeros((nsteps, nfreedof))
velo = np.zeros((nsteps, nfreedof))
acce = np.zeros((nsteps, nfreedof))

glob_disp = np.zeros((nsteps, mod.ndofs))
glob_velo = np.zeros((nsteps, mod.ndofs))
glob_acce = np.zeros((nsteps, mod.ndofs))

F0 = F[0]
acce0 = sp.linalg.solve(M, F[0], assume_a='sym')

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
    

glob_disp[:, mod.free_dof] = disp
glob_velo[:, mod.free_dof] = velo
glob_acce[:, mod.free_dof] = acce

root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
ruta_pickle1 = os.path.join(root, "Trabajo_final_DS", "matrices.pkl")
ruta_pickle2 = os.path.join(root, "Trabajo_final_DS", "resultados.pkl")

matrices = (wk, Phi, M, K, C)
with open(ruta_pickle1, "wb") as f:
    pickle.dump(matrices, f)

resultados = (ttot, glob_disp, glob_velo, glob_acce)
with open(ruta_pickle2, "wb") as f:
    pickle.dump(resultados, f)

    

disp_dof0 = glob_disp.T[0] # grado de libertad en la punta

fig5 = plt.figure()
plt.plot(ttot, disp_dof0, 'b', label='Desplazamiento')
plt.xlabel('Tiempo [s]')
plt.ylabel('Desplazamientos [m]')
plt.legend()
plt.grid()

#---------------------------------------------------------------------------------------------
# Analisis sismico: Derivas

nodos_lado_izq = np.array([135, 126, 117, 108, 100, 92, 83, 74, 67, 60, 52, 45, 
                           38, 28, 23, 17, 12, 7, 5, 2, 1], dtype=int)-1

nodos_lado_der = np.array([143, 141, 137, 128, 120, 111, 98, 90, 84, 76, 69, 55, 
                           51, 44, 39, 33, 25, 21, 19, 16, 15], dtype=int)-1


dofs_li = (np.tile(nodos_lado_der[:,None]*3, 3) + np.arange(3))
dofs_li = dofs_li.astype(int).flatten()[0::3]

disp_stories = glob_disp.T[dofs_li]
drift_stories = np.diff(disp_stories, axis=0)
nstor = disp_stories.shape[0]
stories = np.arange(0, nstor)


#print(disp_stories.shape)
#print(drift_stories.shape)

#mean_drift_stories = np.mean(drift_stories, axis=-1)
#print(mean_drift_stories)

fig6 = plt.figure(figsize=(5,9))

#inst = 2000
drifts_t0 = np.zeros(nstor)
for inst in np.arange(start_sism_idx, end_sism_idx, 100):
    drifts_t0[1:] = drift_stories[:,inst]
    plt.plot(np.abs(drifts_t0)*1000, stories, 
             marker='o', ls='-', label=r'Drifts on {0:.2f}'.format(ttot[inst]))

plt.xlabel('Drift [mm]')
plt.ylabel('Stories')
plt.yticks(stories)
plt.ylim(0,nstor-1)
plt.legend()
plt.grid()
plt.show()
