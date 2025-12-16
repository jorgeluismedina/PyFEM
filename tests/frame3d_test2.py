
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
concrete = Material(elast=2.1e10, poiss=0.25, dense=1.0, constitutive_model="elastic3D") #[N/m2]
materials = [concrete]


C20x30 = RectangularSection(base=0.2, height=0.3, section_direc=0) #[m2]
V20x30 = RectangularSection(base=0.2, height=0.3, section_direc=90) #[m2]

V20x20 = RectangularSection(base=0.2, height=0.2, section_direc=0) #[m2]
sections = [C20x30, V20x30, V20x20]

#'''
coordinates = np.array([[0.0, 0.0, 0.0],
                        [6.0, 0.0, 0.0],
                        [6.0, 6.0, 0.0],
                        [0.0, 6.0, 0.0],

                        [0.0, 0.0, 3.0],
                        [6.0, 0.0, 3.0],
                        [6.0, 6.0, 3.0],
                        [0.0, 6.0, 3.0]])
#'''

#coordinates = np.array([[0.0, 0.0, 0.0],
#                        [0.0, 0.0, 3.0],
#                        [6.0, 0.0, 3.0],
#                        [6.0, 0.0, 0.0]])

# Creacion de Modelo
mod = Model(ndofn=6)
mod.add_nodes(coordinates)
mod.add_materials(materials)
mod.add_sections(sections)

# Añadir elementos
#'''
# Columnas
mod.add_frame_element('Frame3D', concrete, C20x30, [0, 4])
mod.add_frame_element('Frame3D', concrete, C20x30, [1, 5])
mod.add_frame_element('Frame3D', concrete, C20x30, [2, 6])
mod.add_frame_element('Frame3D', concrete, C20x30, [3, 7])
# Vigas 
mod.add_frame_element('Frame3D', concrete, V20x30, [4, 5])
mod.add_frame_element('Frame3D', concrete, V20x30, [5, 6])
mod.add_frame_element('Frame3D', concrete, V20x30, [6, 7])
mod.add_frame_element('Frame3D', concrete, V20x30, [7, 4])
#'''
#mod.add_frame_element('Frame3D', concrete, V20x20, [0, 1])
#mod.add_frame_element('Frame3D', concrete, V20x20, [1, 2])
#mod.add_frame_element('Frame3D', concrete, V20x20, [2, 3])



# Añadir apoyos
#'''
mod.add_node_restraint(0, [1, 1, 1, 1, 1, 1])
mod.add_node_restraint(1, [1, 1, 1, 1, 1, 1])
mod.add_node_restraint(2, [1, 1, 1, 1, 1, 1])
mod.add_node_restraint(3, [1, 1, 1, 1, 1, 1])
#'''

#mod.add_node_restraint(0, [1, 1, 1, 1, 1, 1])
#mod.add_node_restraint(3, [1, 1, 1, 1, 1, 1])



# Añadir cargas nodales
mod.add_node_load(7, [10000, 0, 0, 0, 0, 0]) #[N]
#mod.add_node_load(1, [10000, 0, 0, 0, 0, 0]) #[N]


#mod.set_restraints()
#red = mod.assemb_global_stiff()[np.ix_(mod.free_dof, mod.free_dof)]
#print(mod.elems[0].stiff)



# Solucion
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
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.axis('equal')
plt.show()