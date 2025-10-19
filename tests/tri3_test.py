
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
steel = Material(elast=2.07e8, poiss=0.25, dense=1.0,
                 constitutive_model="plane_stress") #[KN/m2]
materials = [steel]

# Secciones
sect1 = AreaSection(thick=0.013) #[m2]
sections = [sect1]

coordinates = np.array([[0.0, 0.0],
                        [0.0, 1.5],
                        [1.5, 0.0],
                        [1.5, 1.5],
                        [3.0, 0.0],
                        [3.0, 1.5],
                        [4.5, 0.0],
                        [4.5, 1.5]])

# Creacion de Modelo
mod = Model(ndofn=2)
mod.add_nodes(coordinates)
mod.add_materials(materials)
mod.add_sections(sections)

# Añadir elementos
mod.add_area_element('Tri3', steel, sect1, [0, 2, 1])
mod.add_area_element('Tri3', steel, sect1, [1, 2, 3])
mod.add_area_element('Tri3', steel, sect1, [2, 4, 3])
mod.add_area_element('Tri3', steel, sect1, [3, 4, 5])
mod.add_area_element('Tri3', steel, sect1, [4, 6, 5])
mod.add_area_element('Tri3', steel, sect1, [5, 6, 7])

# Añadir apoyos
mod.add_node_restraint(0, [1, 1])
mod.add_node_restraint(1, [1, 1])

# Cargas distribuidas
mod.add_elem_load(1, [[0,0,1], [0,0,0]]) #carga en el tercer lado
mod.add_elem_load(3, [[0,0,1], [0,0,0]])
mod.add_elem_load(5, [[0,0,1], [0,0,0]])

print(mod.elems[1].loads)
print(mod.assemb_global_loads())

#'''

# Solucion
glob_disps, reactions = solve_linear_static(mod)

print('Desplazamientos')
print_matrix(glob_disps*1000, 2, floatfmt=".3e")

mod.calculate_stresses(glob_disps)
print('Esfuerzos Elementos')
for elem in mod.elems:
    print(elem.area)
    print(elem.stress.get_cartesians())
    print(elem.stress.get_principals())
    print(elem.stress.get_von_mises())
#'''

print('Esfuerzos Nodos')
node_cart_stresses, node_prin_stresses, node_vs_stresses = mod.calculate_node_stresses()
print(node_cart_stresses)
print(node_prin_stresses)
print(node_vs_stresses)