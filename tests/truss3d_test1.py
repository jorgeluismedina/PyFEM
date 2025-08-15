
import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import matplotlib.pyplot as plt
from pyfem0.femclass import Model
from pyfem0.plotting import plot_model, print_matrix, plot_3dtruss
from pyfem0.solvers import solve_linear_static

# Materiales
steel = {'elast': 2.0e8} #[KN/m2]
materials = [steel]

# Secciones
sect1 = {'xarea': 0.00139345} #[m2]
sections = [sect1]

# Coordenadas   
coordinates = np.array([[0.0, 0.0, 0.0], 
                        [0.0, 2.0, 0.0], 
                        [2.5, 0.0, 1.7], 
                        [2.5, 2.0, 1.7],
                        [3.5, 1.0, 0.0],
                        [5.5, 1.0, 1.7]])

# Creacion de Modelo
mod = Model(ndofn=3)
mod.add_nodes(coordinates)
mod.add_materials(materials)
mod.add_sections(sections)

# Añadir elementos
mod.add_element(0, [0, 1], sect1, steel, 'Truss3D')
mod.add_element(1, [0, 4], sect1, steel, 'Truss3D')
mod.add_element(2, [1, 4], sect1, steel, 'Truss3D')
mod.add_element(3, [0, 2], sect1, steel, 'Truss3D')
mod.add_element(4, [0, 3], sect1, steel, 'Truss3D')
mod.add_element(5, [1, 3], sect1, steel, 'Truss3D')
mod.add_element(6, [2, 3], sect1, steel, 'Truss3D')
mod.add_element(7, [2, 4], sect1, steel, 'Truss3D')
mod.add_element(8, [3, 4], sect1, steel, 'Truss3D')
mod.add_element(9, [2, 5], sect1, steel, 'Truss3D')
mod.add_element(10, [4, 5], sect1, steel, 'Truss3D')
mod.add_element(11, [3, 5], sect1, steel, 'Truss3D')

# Añadir apoyos
mod.add_node_restraint(0, [1, 1, 1])
mod.add_node_restraint(1, [1, 1, 1])
mod.add_node_restraint(4, [0, 0, 1])
mod.set_restraints()

# Cargas nodales
mod.add_node_load(5, [0.0, 0.0, -100.0])

# Solucion
glob_disps, reactions = solve_linear_static(mod)

print('Desplazamientos')
print_matrix(glob_disps*1000, 2, floatfmt=".3e")

print('Reacciones')
print_matrix(reactions, 2, floatfmt=".3e")

mod.calculate_forces(glob_disps)
print('Fuerzas internas')
print([elem.force for elem in mod.elems])

fig = plot_3dtruss(mod)
fig.show()