

import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import scipy as sp
import pickle

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation


root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
ruta_pickle1 = os.path.join(root, "Trabajo_final_DS", "matrices.pkl")
ruta_pickle2 = os.path.join(root, "Trabajo_final_DS", "resultados.pkl")

with open(ruta_pickle1, "rb") as f:
    matrices = pickle.load(f)

with open(ruta_pickle2, "rb") as f:
    resultados = pickle.load(f)


time, glob_disp, glob_velo, glob_acce = resultados
wk, Phi, M, K, C = matrices


# Desplazamientos
disp_dof0 = glob_disp.T[0] # grado de libertad en la punta

fig5 = plt.figure(figsize=(12,5))
plt.plot(time, disp_dof0, 'b', label='Desplazamiento')
plt.xlabel('Tiempo [s]')
plt.ylabel('Desplazamientos [m]')
plt.legend()
plt.grid()



# Derivas
nodos_lado_izq = np.array([135, 126, 117, 108, 100, 92, 83, 74, 67, 60, 52, 45, 
                           38, 28, 23, 17, 12, 7, 5, 2, 1], dtype=int)-1

nodos_lado_der = np.array([143, 141, 137, 128, 120, 111, 98, 90, 84, 76, 69, 55, 
                           51, 44, 39, 33, 25, 21, 19, 16, 15], dtype=int)-1


dofs_li = (np.tile(nodos_lado_izq[:,None]*3, 3) + np.arange(3))
dofs_li = dofs_li.astype(int).flatten()[0::3]

disp_stories = glob_disp.T[dofs_li]
drift_stories = np.diff(disp_stories, axis=0)
nstor = disp_stories.shape[0]
stories = np.arange(0, nstor, dtype=int)

print(drift_stories.shape)
print(disp_stories.shape)

drifts = np.zeros_like(disp_stories)
drifts[1:] = np.abs(drift_stories) * 1000

max_drifts = np.max(drifts, axis=1)
mean_drifts = np.mean(drifts[:,500:3000], axis=1)


fig6 = plt.figure(figsize=(5,9))
plt.plot(max_drifts, stories, color='blue', marker='o', ls='-', label='max drifts')
plt.plot(mean_drifts, stories, color='red', marker='o', ls='-', label='mean drifts')
plt.xlabel('Drift [mm]')
plt.ylabel('Stories')
plt.yticks(stories)
plt.xlim(0, 16)
plt.ylim(0, nstor-1)
plt.legend(loc='upper right', fancybox=False, edgecolor='black')
plt.grid()
plt.show()


# Animacion 

'''
drifts_ti = drifts.T[400:3500]
time_sis = time[400:3500]
N = drifts_ti.shape[0]


dur = 25
fps = 25
max_frames = int(dur*fps)

indices = np.linspace(0, N-1, max_frames).astype(int)
drifts_ti_red = drifts_ti[indices]
time_sis_red = time_sis[indices]


def animate(i):
    ln1.set_data(drifts_ti_red[i], stories)
    text.set_text('t = {:.2f} [s]'.format(time_sis_red[i]))
    return ln1, text

fig, ax = plt.subplots(1,1, figsize=(5,9))
ln1, = plt.plot([], [], marker='o', ls='-')#, makersize=8)
text = plt.text(10, 19, 'asdasd', fontsize=15, backgroundcolor='white', ha='right')
ax.set_xlabel('Drift [mm]')
ax.set_ylabel('Stories')
ax.set_xlim(0, 15)
ax.set_ylim(0, nstor-1)
ax.grid()

ani = animation.FuncAnimation(fig, animate, frames=len(indices), interval=1000/fps)
ani.save('drifts.gif',writer='pillow',fps=fps)
'''