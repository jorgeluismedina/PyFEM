
import sys
import os
# Añadir el directorio raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from pyfem.femclass import Model
from pyfem.materials.material import Material
from pyfem.sections import AreaSection
from pyfem.plotting import print_matrix, plot_2dmodel
from pyfem.solvers import solve_linear_static

# Materiales
steel = Material(elast=2.07e8, poiss=0.25, dense=1.0) #[KN/m2]
materials = [steel]

# Secciones
sect1 = AreaSection(thick=0.013) #[m2]
sections = [sect1]

coordinates = np.array([[0.075, 0.000],
                        [0.075, 0.050],
                        [0.000, 0.050],
                        [0.000, 0.000]])

# Creacion de Modelo
mod = Model(ndofn=2)
mod.add_nodes(coordinates)
mod.add_materials(materials)
mod.add_sections(sections)

# Añadir elementos
mod.add_element(0, [0, 1, 3], sect1, steel, 'Tri3')
mod.add_element(1, [3, 1, 2], sect1, steel, 'Tri3')

# Añadir apoyos
mod.add_node_restraint(0, [0, 1])
mod.add_node_restraint(2, [1, 1])
mod.add_node_restraint(3, [1, 1])

# Cargas nodales
mod.add_node_load(1, [0.0, -4.45]) #[KN]

#elem2 = mod.elems[1]
#print(elem2.quad_scheme.points)
#print(elem2.coord)
#print_matrix(elem2.stiff, 2, floatfmt=".3e")

#'''
mod.set_restraints()
Kglob = mod.assemb_global_stiff()
print_matrix(mod.assemb_global_stiff(), 2, floatfmt=".3e")
#print_matrix(Kglob[np.ix_(mod.free_dof, mod.free_dof)], 2, floatfmt=".3e")

# Solucion
glob_disps, reactions = solve_linear_static(mod)

print('Desplazamientos')
print_matrix(glob_disps*1000, 2, floatfmt=".3e")

mod.calculate_stresses(glob_disps)
print('Esfuerzos')
for elem in mod.elems:
    print(elem.stress)
#'''

node_stresses = mod.calculate_node_stresses()
print(node_stresses)