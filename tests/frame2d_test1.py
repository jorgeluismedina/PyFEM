
import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import matplotlib.pyplot as plt
from pyfem.femclass import Model
from pyfem.materials.material import Material
from pyfem.sections import FrameSection
from pyfem.plotting import plot_2dmodel, print_matrix
from pyfem.solvers import solve_linear_static

# Materiales
steel = Material(elast=2.1e8, poiss=0.2, dense=1.0) #[KN/m2]
materials = [steel]

# Secciones
sect1 = FrameSection(xarea=0.00421, inrt3=8.358e-5) #[m2]
sections = [sect1]

# Coordenadas
coordinates = np.array([[0.0, 0.0], 
                        [0.0, 3.0], 
                        [3.0, 3.0], 
                        [6.0, 3.0],
                        [9.0, 3.0]]) 


# Creacion de Modelo
mod = Model(ndofn=3)
mod.add_nodes(coordinates)
mod.add_materials(materials)
mod.add_sections(sections)

# Añadir elementos
mod.add_element(0, [0, 2], sect1, steel, 'Frame22D')
mod.add_element(1, [1, 2], sect1, steel, 'Frame22D')
mod.add_element(2, [2, 3], sect1, steel, 'Frame22D')
mod.add_element(3, [3, 4], sect1, steel, 'Frame22D')

# Añadir apoyos
mod.add_node_restraint(0, [1, 1, 0])
mod.add_node_restraint(1, [1, 1, 0])
mod.add_node_restraint(4, [1, 0, 1])

# Cargas distribuidas
q = -10 #[KN/m]
mod.add_elem_load(3, [0.0, q, 0.0, q])

# Realeses de elementos
mod.elems[0].release_ends(ri=1, rj=0)
mod.elems[2].release_ends(ri=1, rj=0)

# Solucion
glob_disps, reactions = solve_linear_static(mod)

# Mostrar Resultados
unit_changer = np.tile(np.array([1000, 1000, 1]), mod.nnods)
print_matrix(glob_disps*unit_changer, 2, floatfmt=".3e")
print(reactions)


fig1 = plot_2dmodel(mod)
plt.show()