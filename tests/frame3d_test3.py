
import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from pyfem.femclass import Model
from pyfem.materials.material import Material
from pyfem.sections import RectangularSection
from pyfem.solvers import solve_linear_static


# Materiales
E = 4700 * np.sqrt(20) * 1000 * 1000
concrete = Material(elast=E, poiss=0.20, dense=1.0,
                    constitutive_model="elastic3D") #[N/m2]
materials = [concrete]

V25x50 = RectangularSection(base=0.25, height=0.50, section_direc=90) #[m2]
v10x20 = RectangularSection(base=0.10, height=0.20, section_direc=90) #[m2]
sections = [V25x50, v10x20]

coordinates = np.array([
    [0.000, 0.000, 0.000],
    [0.128, 0.866, 0.000],
    [0.500, 0.000, 0.000],
    [0.500, 1.658, 0.000],
    [1.000, 0.000, 0.000],
    [1.000, 2.236, 0.000],
    [1.500, 0.000, 0.000],
    [1.500, 2.598, 0.000],
    [2.000, 0.000, 0.000],
    [2.000, 2.828, 0.000],
    [2.500, 0.000, 0.000],
    [2.500, 2.828, 0.000],
    [3.000, 0.000, 0.000],
    [3.000, 2.828, 0.000],
    [3.500, 0.000, 0.000],
    [3.500, 2.958, 0.000],
    [4.000, 0.000, 0.000],
    [4.000, 2.828, 0.000],
    [4.500, 0.000, 0.000],
    [4.500, 2.598, 0.000],
    [5.000, 0.000, 0.000],
    [5.000, 2.236, 0.000],
    [5.500, 0.000, 0.000],
    [5.500, 1.658, 0.000],
    [5.872, 0.866, 0.000],
    [6.000, 0.000, 0.000]
])

conectivities = np.array([
    [1, 2],
    [2, 4],
    [4, 6],
    [6, 8],
    [8, 10],
    [10, 12],
    [12, 14],
    [14, 16],
    [16, 18],
    [18, 20],
    [20, 22], 
    [22, 24],
    [24, 25],
    [25, 26],

    [3, 4],
    [5, 6], 
    [7, 8],
    [9, 10],
    [11, 12],
    [13, 14],
    [15, 16],
    [17, 18],
    [19, 20],
    [21, 22], 
    [23, 24]
]) - 1

# Creacion de Modelo
mod = Model(ndofn=6)
mod.add_nodes(coordinates)
mod.add_materials(materials)
mod.add_sections(sections)

# Añadir elementos
for nc in range(14):
    mod.add_frame_element('Frame3D', concrete, V25x50, conectivities[nc])

for nc in range(14, 25):
    mod.add_frame_element('Frame3D', concrete, v10x20, conectivities[nc])



# Añadir apoyos
restricted_nodes = np.hstack([np.arange(1,24,2),26])

for rn in restricted_nodes:
    mod.add_node_restraint(rn, [1, 1, 1, 1, 1, 1])

#mod.add_node_restraint(0, [1, 1, 1, 1, 1, 1])

# Añadir cargas nodales
mod.add_node_load(7, [10000, 0, 0, 0, 0, 0]) #[N]


glob_disps, reactions = solve_linear_static(mod)
print(glob_disps.reshape((mod.nnods, mod.ndofn)), '\n')


fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})

for elem in mod.elems:
    ax.plot(*elem.coord.T, color='b')
    ax.quiver(*elem.mid_point, *elem.local_axes[0], length=0.7, arrow_length_ratio=0.2, color='red')#colors=['red', 'lime', 'cyan']
    ax.quiver(*elem.mid_point, *elem.local_axes[1], length=0.7, arrow_length_ratio=0.2, color='lime')
    ax.quiver(*elem.mid_point, *elem.local_axes[2], length=0.7, arrow_length_ratio=0.2, color='cyan')

for n, coor in enumerate(mod.coord):
    ax.text(coor[0]+0.05, coor[1]+0.05, coor[2]+0.05, str(n), 
            fontsize=7, ha='left', va='bottom')

ax.xaxis.set_pane_color('white')  # Transparent x-pane
ax.axis('equal')
plt.show()