
import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from pyfem.femclass import Model
from pyfem.materials.material import Material
from pyfem.sections import AreaSection
from pyfem.solvers import solve_linear_static


# Materiales
concrete = Material(elast=2.57e7, poiss=0.2, dense=0.0,
                    constitutive_model="plane_stress") #[KN/m2]
materials = [concrete]

# Secciones
sect1 = AreaSection(thick=0.2) #[m2]
sections = [sect1]

l = 9
h = 3
l_divitions = 6
h_divitions = 2
l_nnods = l_divitions + 1
h_nnods = h_divitions + 1
x = np.linspace(0, l, l_nnods)
y = np.linspace(0, h, h_nnods)
X, Y = np.meshgrid(x,y)

coordinates = np.column_stack((X.ravel(), Y.ravel()))
idx_nodes = np.arange(coordinates.shape[0]).reshape((h_nnods, l_nnods))
print(idx_nodes)



# Creacion de Modelo
mod = Model(ndofn=2)
mod.add_nodes(coordinates)
mod.add_materials(materials)
mod.add_sections(sections)

# Anadir Elementos
for i in range(h_divitions):
    for j in range(l_divitions):
        elem_grid = idx_nodes[i:i+2, j:j+2]
        conec = [elem_grid[0,0], elem_grid[0,1], elem_grid[1,1], elem_grid[1,0]]
        #print(conec)
        mod.add_area_element('Quad4', concrete, sect1, conec)

# Añadir restricciones
fixed_nodes = idx_nodes[:,0]
#print(fixed_nodes)
fixed_conditions = [1,1]
for n in fixed_nodes:
    mod.add_node_restraint(n, fixed_conditions)

# Añadir fuerzas nodales
mod.add_node_load(13, [0,-100])


# Resolucion del sistema 
glob_disps, reactions = solve_linear_static(mod)
node_disps = glob_disps.reshape((mod.nnods, mod.ndofn))

esc = 500
def_coords = coordinates + node_disps * esc



fig, ax = plt.subplots(1,1)

for i, coord in enumerate(def_coords):
    ax.text(coord[0]+0.05, coord[1]+0.05, str(i), fontsize=7, ha='left', va='bottom')

ax.scatter(*def_coords.T, c='b', s=2.0)
for elem in mod.elems:
    def_ecoord = def_coords[elem.conec]
    plot_def_ecoord = np.vstack([def_ecoord, def_ecoord[0]])
    ax.plot(*plot_def_ecoord.T, 'b-', linewidth=1.0, alpha=0.8)

ax.axis("equal")
plt.show()

