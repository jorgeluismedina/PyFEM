
import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from pyfem.femclass import Model
from pyfem.materials.material import Material
from pyfem.plotting import print_matrix
from pyfem.solvers import solve_linear_static
from pyfem.gauss_quad import gauss_nd

# Materiales
steel = Material(elast=2.07e8, poiss=0.25, dense=1.0,
                 constitutive_model="elastic3D") #[KN/m2]
materials = [steel]

coordinates = np.array([[0.0, 1.5, 0.0],
                        [1.5, 1.5, 0.0],
                        [1.5, 1.5, 1.5],
                        [0.0, 1.5, 1.5],
                        [0.0, 0.0, 0.0],
                        [1.5, 0.0, 0.0],
                        [1.5, 0.0, 1.5],
                        [0.0, 0.0, 1.5],
                        [3.0, 1.5, 0.0],
                        [3.0, 1.5, 1.5],
                        [3.0, 0.0, 0.0],
                        [3.0, 0.0, 1.5]])

# Creacion de Modelo
mod = Model(ndofn=3)
mod.add_nodes(coordinates)
mod.add_materials(materials)

#print(gauss_nd(2, 3))
#'''
# Añadir elementos
mod.add_solid_element('Hex8', steel, [0, 1, 2, 3, 4, 5, 6, 7])
mod.add_solid_element('Hex8', steel, [1, 8, 9, 2, 5, 10, 11, 6])

# Añadir apoyos
mod.add_node_restraint(0, [1, 1, 1])
mod.add_node_restraint(4, [1, 1, 1])
mod.add_node_restraint(7, [1, 1, 1])
mod.add_node_restraint(3, [1, 1, 1])

# Cargas nodales
mod.add_node_load(9,  [0, 0, -1000])
mod.add_node_load(11, [0, 0, -1000])

print(mod.elems[0].volume, mod.elems[1].volume)
print(mod.elems[0].stiff.shape, mod.elems[1].stiff.shape)
print(mod.elems[0].bload, mod.elems[1].bload)
#'''
# Solucion del sistema
glob_disps1, reactions1 = solve_linear_static(mod)

#print('Desplazamientos')
#print_matrix(glob_disps1*1000, 2, floatfmt=".3e")


# Calculo de esfuerzos
mod.calculate_stresses(glob_disps1)
#print(mod.elems[0].stress, mod.elems[1].stress)

new_coords1 = mod.coord + 2000 * glob_disps1.reshape((mod.nnods, 3))
#print(new_coords1.shape)


'''
# Elementos Enriquecidos
mod2 = Model(ndofn=3)
mod2.add_nodes(coordinates)
mod2.add_materials(materials)


# Añadir elementos
mod2.add_solid_element('Hex8E', steel, [0, 1, 2, 3, 4, 5, 6, 7])
mod2.add_solid_element('Hex8E', steel, [1, 8, 9, 2, 5, 10, 11, 6])

# Añadir apoyos
mod2.add_node_restraint(0, [1, 1, 1])
mod2.add_node_restraint(4, [1, 1, 1])
mod2.add_node_restraint(7, [1, 1, 1])
mod2.add_node_restraint(3, [1, 1, 1])

# Cargas nodales
mod2.add_node_load(9,  [0, 0, -1000])
mod2.add_node_load(11, [0, 0, -1000])

print(mod2.elems[0].volume, mod.elems[1].volume)
print(mod2.elems[0].stiff.shape, mod.elems[1].stiff.shape)
print(mod2.elems[0].bload, mod.elems[1].bload)

# Solucion del sistema
glob_disps2, reactions2 = solve_linear_static(mod2)
new_coords2 = mod2.coord + 2000 * glob_disps2.reshape((mod2.nnods, 3))
#print(new_coords1.shape)

# Calculo de esfuerzos
mod2.calculate_stresses(glob_disps2)

'''

#plt.figure(1)
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
ax.scatter(*mod.coord.T, color='blue', label='undeformed')
ax.scatter(*new_coords1.T, color='red', label='deformed')
ax.axis('equal')
ax.legend()

#plt.figure(2)
#fig, ax = plt.subplots(projection='3d')
#ax.scatter(*mod2.coord.T, color='blue', label='undeformed')
#ax.scatter(*new_coords2.T, color='red', label='deformed')
#ax.axis('equal')
#ax.legend()

plt.show()
