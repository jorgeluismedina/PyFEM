

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
ruta_pickle1 = os.path.join(root, "Trabajo_final_DS", "sismos.pkl")
ruta_pickle2 = os.path.join(root, "Trabajo_final_DS", "matrices.pkl")
ruta_pickle3 = os.path.join(root, "Trabajo_final_DS", "resultados1.pkl") #Stiff rojo
ruta_pickle4 = os.path.join(root, "Trabajo_final_DS", "resultados2.pkl") #Soft azul

with open(ruta_pickle1, "rb") as f:
    sismos = pickle.load(f)

with open(ruta_pickle2, "rb") as f:
    matrices = pickle.load(f)

with open(ruta_pickle3, "rb") as f:
    resultados1 = pickle.load(f)

with open(ruta_pickle4, "rb") as f:
    resultados2 = pickle.load(f)


ag1, ag2 = sismos
time, glob_disp1, glob_velo1, glob_acce1 = resultados1 #Stiff rojo
time, glob_disp2, glob_velo2, glob_acce2 = resultados2 #Soft azul
wk, Phi, M, K, C = matrices

# indices del sismo
start = 500
end = 3000


# Desplazamientos
topdisp1 = glob_disp1.T[0] # grado de libertad en la punta
topdisp2 = glob_disp2.T[0] # grado de libertad en la punta
topacce1 = glob_acce1.T[0] # grado de libertad en la punta
topacce2 = glob_acce2.T[0] # grado de libertad en la punta

print('max last floor disp stiff: {0:.4f} cm'.format(np.max(np.abs(topdisp1))*100))
print('max last floordisp soft: {0:.4f} cm'.format(np.max(np.abs(topdisp2))*100))
print('max last floor acce stiff: {0:.4f} m/s2'.format(np.max(np.abs(topacce1))))
print('max last floor acce soft: {0:.4f} m/s2'.format(np.max(np.abs(topacce2))))

fig1, ax1 = plt.subplots(2,1)
ax1[0].plot(time, topdisp1, 'r', label='Stiff soil displacement')
ax1[0].set_xlabel('Time [s]')
ax1[0].set_ylabel('Displacement [m]')
ax1[0].axvline(x=5, color='black', ls='--')
ax1[0].axvline(x=25, color='black', ls='--')
ax1[0].set_xlim(0, 50)
ax1[0].legend(loc='upper right', fancybox=False, edgecolor='black')
ax1[0].grid()
ax1[1].plot(time, topdisp2, 'b', label='Soft soil displacement')
ax1[1].set_xlabel('Time [s]')
ax1[1].set_ylabel('Displacement [m]')
ax1[1].axvline(x=5, color='black', ls='--')
ax1[1].axvline(x=30, color='black', ls='--')
ax1[1].set_xlim(0, 50)
ax1[1].legend(loc='upper right', fancybox=False, edgecolor='black')
ax1[1].grid()

fig2, ax2 = plt.subplots(2,1)
ax2[0].plot(time, topacce1, 'r', label='Stiff soil acceleration')
ax2[0].set_xlabel('Time [s]')
ax2[0].set_ylabel('Acceleration [m/s2]')
ax2[0].axvline(x=5, color='black', ls='--')
ax2[0].axvline(x=25, color='black', ls='--')
ax2[0].set_xlim(0, 50)
ax2[0].legend(loc='upper right', fancybox=False, edgecolor='black')
ax2[0].grid()
ax2[1].plot(time, topacce2, 'b', label='Soft soil acceleration')
ax2[1].set_xlabel('Time [s]')
ax2[1].set_ylabel('Acceleration [m/s2]')
ax2[1].axvline(x=5, color='black', ls='--')
ax2[1].axvline(x=30, color='black', ls='--')
ax2[1].set_xlim(0, 50)
ax2[1].legend(loc='upper right', fancybox=False, edgecolor='black')
ax2[1].grid()


#--------------------------------------------------------------------------------------------------------
nodos_lado_izq = np.array([135, 126, 117, 108, 100, 92, 83, 74, 67, 60, 52, 45, 
                           38, 28, 23, 17, 12, 7, 5, 2, 1], dtype=int)-1

nodos_lado_der = np.array([143, 141, 137, 128, 120, 111, 98, 90, 84, 76, 69, 55, 
                           51, 44, 39, 33, 25, 21, 19, 16, 15], dtype=int)-1


dofs_li = (np.tile(nodos_lado_izq[:,None]*3, 3) + np.arange(3))
dofs_li = dofs_li.astype(int).flatten()[0::3]
dofs_ld = (np.tile(nodos_lado_der[:,None]*3, 3) + np.arange(3))
dofs_ld = dofs_ld.astype(int).flatten()[0::3]


# Sismo
fig2, ax2 = plt.subplots(2,1)
ax2[0].plot(time[start:end], ag1[start:end], color='red', label='stiff soil')
ax2[0].set_xlabel('Time [s]')
ax2[0].set_ylabel('Ground Acceleration [m/s2]')
ax2[0].legend(loc='upper right', fancybox=False, edgecolor='black')
ax2[0].grid()
ax2[1].plot(time[start:end], ag2[start:end], color='blue', label='soft soil')
ax2[1].set_xlabel('Time [s]')
ax2[1].set_ylabel('Ground Acceleration [m/s2]')
ax2[1].legend(loc='upper right', fancybox=False, edgecolor='black')
ax2[1].grid()

#-----------------------------------------------------------------------------------------------------
nstor = len(nodos_lado_izq)
stories = np.arange(0, nstor, dtype=int)

# Desplazamientos #(solo lado derecho) 
disp_stories_right1 = glob_disp1.T[dofs_ld]
disp_stories_right2 = glob_disp2.T[dofs_ld]

max_disp_right1 = np.max(disp_stories_right1, axis=1)
mean_disp_right1 = np.mean(disp_stories_right1[:,start:end], axis=1)
max_disp_right2 = np.max(disp_stories_right2, axis=1)
mean_disp_right2 = np.mean(disp_stories_right2[:,start:end], axis=1)

fig5 = plt.figure(figsize=(5,9))
plt.plot(max_disp_right1*100, stories, color='red', marker='s', ls='-', label='max disp stiff')
#plt.plot(mean_disp_right1*100, stories, color='red', marker='o', ls='--', label='mean disp left')
plt.plot(max_disp_right2*100, stories, color='blue', marker='s', ls='-', label='max disp soft')
#plt.plot(mean_disp_right2*100, stories, color='blue', marker='o', ls='--', label='mean disp right')
plt.xlabel('Displacements [cm]')
plt.ylabel('Stories')
plt.yticks(stories)
plt.xlim(0, 30)
plt.ylim(0, nstor-1)
plt.legend(loc='lower right', fancybox=False, edgecolor='black')
plt.grid()


# Derivas (solo lado derecho)
#drifts_left = np.zeros_like(disp_stories_left)
#drifts_left[1:] = np.abs(np.diff(disp_stories_left, axis=0) / 3.15)
drifts_right1 = np.zeros_like(disp_stories_right1)
drifts_right1[1:] = np.abs(np.diff(disp_stories_right1, axis=0) / 3.15)
drifts_right2 = np.zeros_like(disp_stories_right2)
drifts_right2[1:] = np.abs(np.diff(disp_stories_right2, axis=0) / 3.15)

#max_drifts_left = np.max(drifts_left, axis=1)
#mean_drifts_left = np.mean(drifts_left[:,start:end], axis=1)
max_drifts_right1 = np.max(drifts_right1, axis=1)
mean_drifts_right1 = np.mean(drifts_right1[:,start:end], axis=1)
max_drifts_right2 = np.max(drifts_right2, axis=1)
mean_drifts_right2 = np.mean(drifts_right2[:,start:end], axis=1)

print('max drift stiff: {0:.4f}'.format(np.max(max_drifts_right1)))
print('max drift soft: {0:.4f}'.format(np.max(max_drifts_right2)))

fig6 = plt.figure(figsize=(5,9))
plt.plot(max_drifts_right1, stories, color='red', marker='s', ls='-', label='max drifts stiff')
plt.plot(mean_drifts_right1, stories, color='red', marker='o', ls='--', label='mean drifts stiff')
plt.plot(max_drifts_right2, stories, color='b', marker='s', ls='-', label='max drifts soft')
plt.plot(mean_drifts_right2, stories, color='b', marker='o', ls='--', label='mean drifts soft')
plt.xlabel('Drift (unitless)')
plt.ylabel('Stories')
plt.yticks(stories)
plt.xlim(0, 0.007)
plt.ylim(0, nstor-1)
plt.legend(loc='upper right', fancybox=False, edgecolor='black')
plt.grid()
plt.show()


# Animacion 

#'''
drifts_ti1 = drifts_right1.T[start-200:end+600]
drifts_ti2 = drifts_right2.T[start-200:end+600]
time_sis = time[start-200:end+600]
N = drifts_ti1.shape[0]


dur = 25
fps = 25
max_frames = int(dur*fps)

indices = np.linspace(0, N-1, max_frames).astype(int)
drifts_ti_red1 = drifts_ti1[indices]
drifts_ti_red2 = drifts_ti2[indices]
time_sis_red = time_sis[indices]


fig3, ax3 = plt.subplots(1,2, figsize=(10,9))
ln1, = ax3[0].plot([], [], marker='o', color='r', ls='-', label='stiff soil')#, makersize=8)
#text = ax3[0].text(0.65, 0.95, 'asdasd', transform=ax3[0].transAxes, va='top', 
#                   fontsize=15, backgroundcolor='white')
ln2, = ax3[1].plot([], [], marker='o', color='b', ls='-', label='soft soil')#, makersize=8)
text = ax3[1].text(0.55, 0.05, 'asdasd', transform=ax3[1].transAxes, va='top', 
                   fontsize=15, backgroundcolor='white')


ax3[0].set_xlabel('Drift')
ax3[0].set_ylabel('Stories')
ax3[0].set_xlim(0, 0.007)
ax3[0].set_ylim(0, nstor-1)
ax3[0].set_yticks(stories)
ax3[0].legend(loc='upper right', fancybox=False, edgecolor='black')
ax3[0].grid()

ax3[1].set_xlabel('Drift')
ax3[1].set_ylabel('Stories')
ax3[1].set_xlim(0, 0.007)
ax3[1].set_ylim(0, nstor-1)
ax3[1].set_yticks(stories)
ax3[1].legend(loc='upper right', fancybox=False, edgecolor='black')
ax3[1].grid()

def animate(i):
    ln1.set_data(drifts_ti_red1[i], stories)
    ln2.set_data(drifts_ti_red2[i], stories)
    text.set_text('t = {:.2f} [s]'.format(time_sis_red[i]))
    return ln1, ln2, text



ani = animation.FuncAnimation(fig3, animate, frames=len(indices), interval=1000/fps)
ani.save('drifts.gif',writer='pillow',fps=fps)
#'''